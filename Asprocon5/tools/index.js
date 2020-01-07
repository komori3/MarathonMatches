// Visualizer Ver.13
// JavaScript source code
"use strict"

class FileParser {
	constructor(filename, content) {
		this.filename = filename;
		this.content = [];
		for (let _i = 0, _a = content.split('\n'); _i < _a.length; _i++) {
			let line = _a[_i];
			let words = line.trim().split(new RegExp('\\s+'));
			this.content.push(words);
		}
		this.y = 0;
		this.x = 0;
	}

	isEOF() {
		return this.content.length <= this.y;
	}

	getWord() {
		if (this.isEOF()) {
			this.reportError('a word expected, but EOF');
		}
		if (this.content[this.y].length <= this.x) {
			this.reportError('a word expected, but newline');
		}
		let word = this.content[this.y][this.x];
		this.x += 1;
		return word;
	}

	getInt() {
		let word = this.getWord();
		if (!word.match(new RegExp('^[-+]?[0-9]+$'))) {
			this.reportError("a number expected, but word " + JSON.stringify(this.content[this.y][this.x]));
		}
		return parseInt(word);
	}

	getNewline() {
		if (this.isEOF()) {
			this.reportError('newline expected, but EOF');
		}
		this.x = 0;
		this.y += 1;
	}

	unWind() {
		if (this.x == 0) {
			this.y -= 1;
			this.x = this.content[this.y].length - 1;
		} else {
			this.x -= 1;
		}
	}

	reportError(msg) {
		msg = this.filename + ": line " + (this.y + 1) + ": " + msg;
		alert(msg);
		throw new Error(msg);
	}

}

function loadFile(file, callback) {
	let reader = new FileReader();
	reader.readAsText(file);
	reader.onloadend = function () {
		callback(reader.result);
	};
}

class FileSelector {
	constructor() {
		let _this = this;
		this.inputFile = document.getElementById("inputFile");
		this.outputFile = document.getElementById("outputFile");
		this.reloadButton = document.getElementById("reloadButton");
		this.reloadFilesClosure = function () { _this.reloadFiles(); };
		this.inputFile.addEventListener('change', this.reloadFilesClosure);
		this.outputFile.addEventListener('change', this.reloadFilesClosure);
		this.reloadButton.addEventListener('click', this.reloadFilesClosure);
	}

	reloadFiles() {
		let _this = this;
		if (this.inputFile.files == null || this.inputFile.files.length == 0)
			return;
		loadFile(this.inputFile.files[0], function (inputContent) {
			if (_this.outputFile.files == null || _this.outputFile.files.length == 0)
				return;
			loadFile(_this.outputFile.files[0], function (outputContent) {
				_this.inputFile.removeEventListener('change', _this.reloadFilesClosure);
				_this.outputFile.removeEventListener('change', _this.reloadFilesClosure);
				_this.reloadButton.classList.remove('disabled');
				if (_this.callback !== undefined) {
					_this.callback(inputContent, outputContent);
				}
			});
		});
	}
}

class InputFile {
	constructor(content) {
		let parser = new FileParser('<input-file>', content.trim());
		parser.getWord();
		this.I = parser.getInt();
		this.M = parser.getInt();
		this.MM = parser.getInt();
		this.BL = parser.getInt();
		this.CL = parser.getInt();
		this.R = parser.getInt();
		parser.getNewline();

		this.readItem(parser);
		this.readBom(parser);
		this.readCombi(parser);
		this.readOrder(parser);
        
		if (!parser.isEOF()) {
			parser.reportError("Too long file.");
		}
	}

	readItem(parser){
		this.item = [];
		for (let i = 0; i < this.I; i++) {
			parser.getWord();
			let _i = parser.getInt();
			let s = parser.getInt();
			this.item[_i] = s;
			parser.getNewline();
		}
	}

	readBom(parser) {
		this.boms = [];
		for (let i = 0; i < this.BL; ++i) {
			parser.getWord();
			let _i = parser.getInt();
			let s = parser.getInt();
			let m = parser.getInt();
			let c = parser.getInt();
			let x = parser.getInt();
			let y = parser.getInt();

			if (this.boms[_i] == undefined) {
				this.boms[_i] = [];
			}
			if (this.boms[_i][m] == undefined) {
				this.boms[_i][m] = {};
			}
			this.boms[_i][m].s = s;
			this.boms[_i][m].c = c;
			this.boms[_i][m].x = x;
			this.boms[_i][m].y = y;

			
			parser.getNewline();
		}
	}

	readCombi(parser) {
		this.combi = [];
		for (let m = 0; m < this.MM; ++m) {
			this.combi[m] = {};
			this.combi[m] = [];
			for (let ipre = 0; ipre < this.I; ++ipre) {
				this.combi[m][ipre] = {};
				this.combi[m][ipre] = [];
			}
		}
		for (let i = 0; i < this.CL; ++i) {
			parser.getWord();
			let m = parser.getInt();
			let ipre = parser.getInt();
			let inext = parser.getInt();
			let t = parser.getInt();
			this.combi[m][ipre][inext] = t;
			parser.getNewline();
		}
	}

	readOrder(parser) {
		this.orders = [];
		for (let i = 0; i < this.R; ++i) {
			parser.getWord();
			let r = parser.getInt();
			let _i = parser.getInt();
			let e = parser.getInt();
			let d = parser.getInt();
			let q = parser.getInt();
			let pr = parser.getInt();
			let a = parser.getInt();
			this.orders[r] = {};
			this.orders[r].i = _i;
			this.orders[r].e = e;
			this.orders[r].d = d;
			this.orders[r].q = q;
			this.orders[r].pr = pr;
			this.orders[r].a = a;
			parser.getNewline();
		}
	}
}

class OutputFile {
	constructor(content, inputFile) {
		this.readFile(content, inputFile);

		for (let r = 0; r < inputFile.R; ++r) {
			this.operations.sort(function (a, b) {
				if (a.t1 > b.t1) {
					return 1;
				} else {
					return -1;
				}
			});
		}

		let mToPreviousT3 = [];
		let mToPreviousI = [];
		let rpToTime = [];

		for (let r = 0; r < inputFile.R; ++r) {
			let operation = this.operations[r];
			let order = inputFile.orders[operation.r];
			let i = order.i;
			let _m = operation.m;

			rpToTime[operation.r] = {};
			let m = _m[0];		
			if (inputFile.boms[i][m] != undefined) {
				if (mToPreviousT3[m] == undefined) mToPreviousT3[m] = 0;

				let t;
				if (mToPreviousI[m] == undefined) {
					t = 0;
				} else if (inputFile.combi[m][mToPreviousI[m]][i] != undefined) {
					t = inputFile.combi[m][mToPreviousI[m]][i];
				} else {
					t = 0;
				}
				let t1 = Math.max(mToPreviousT3[m], order.e, operation.t1);
				let needResourceAmount = t;
				let t2 = t1 + needResourceAmount;

				needResourceAmount = order.q * inputFile.boms[i][m].c;
				let t3 = t2 + needResourceAmount;

				rpToTime[operation.r].valid = true;
				rpToTime[operation.r].t1 = t1;
				rpToTime[operation.r].t2 = t2;
				rpToTime[operation.r].t3 = t3;
				mToPreviousI[m] = i;
				mToPreviousT3[m] = t3;
			} else {
				rpToTime[operation.r].valid = false;
				rpToTime[operation.r].t1 = 0;
				rpToTime[operation.r].t2 = 0;
				rpToTime[operation.r].t3 = 0;
			}
		}

		this.errorCheck(inputFile, rpToTime);

		this.max_t = 0;
		for (let r = 0; r < inputFile.R; ++r) {
			this.max_t = Math.max(this.max_t, this.operations[r].t3, inputFile.orders[r].d);
		}
	}

	errorCheck(inputFile, rpToTime) {
		for (let r = 0; r < inputFile.R; ++r) {
			this.operations.sort(function (a, b) {
				if (a.r > b.r) {
					return 1;
				} else {
					return -1;
				}
			});
		}
		let judge = false;
		for (let r = 0; r < inputFile.R; ++r) {
			let str = "\nオーダ番号: " + r;
			let ope = this.operations[r];
			for (let j = 0; j < inputFile.item[ope.i]; j++) {
				if (rpToTime[r].valid && ope.m[j] != -1) {
					if (boms[ope.i][ope.m[j]].s !== j) {
						alert("役割間違い" + str);
						judge = true;
					}
					if (rpToTime[r].t1 > ope.t1) {
						alert("製造開始時刻違反" + str);
						judge = true;
					}
					if ((rpToTime[r].t2 - rpToTime[r].t1) * inputFile.boms[ope.i][ope.m[j]].x != ope.t2 - ope.t1) {
						alert("段取り時間間違い" + str);
						judge = true;
					}
					if ((rpToTime[r].t3 - rpToTime[r].t2) * inputFile.boms[ope.i][ope.m[j]].y != this.operations[r].t3 - this.operations[r].t2) {
						alert("製造時間間違い" + str);
						judge = true;
					}
				}
				else if (ope.m[j] == -1) {
					if (inputFile.boms[ope.i][ope.m[j]].y != 0 || inputFile.boms[ope.i][ope.m[j]].x * (ope.t1 - ope.t2) != 0) {
						alert("資源未割り付け" + str);
						judge = true;
					}
				}
				else {
					alert("BOM違反" + str);
					judge = true;
				}
			}
			if (judge) break;
		}
		return judge;
	}

	readFile(content, inputFile) {
		let parser = new FileParser('<output-file>', content.trim());
		this.operations = [];
		for (let i = 0; i < inputFile.R; ++i) {
			let r = parser.getInt();
			let t1 = parser.getInt();
			let t2 = parser.getInt();
			let t3 = parser.getInt();


			if (this.operations[r] == undefined) {
				this.operations[r] = {};
				this.operations[r].m = [];
			}
			this.operations[r].r = r;
			this.operations[r].t1 = t1;
			this.operations[r].t2 = t2;
			this.operations[r].t3 = t3;
			for (let s = 0; s < inputFile.item[inputFile.orders[r].i]; s++) {
				let m = parser.getInt();
				this.operations[r].m[s] = m;
			}
			this.operations[r].valid = true;
			parser.getNewline();
		}

		if (!parser.isEOF()) {
			parser.reportError("Too long file.");
		}
	}
}

class TesterFrame {
	constructor(input, output) {
		this.input = input;
		this.output = output;

		this.grossP = 10000000000;
		this.dcost = 0.0;
		this.scost = 0.0;
		this.profit = 0.0;

		for (let r = 0; r < input.R; ++r) {
			let d = input.orders[r].d;
			let a = input.orders[r].a;
			let pr = input.orders[r].pr;
			let i = input.orders[r].i;
			let t1 = output.operations[r].t1;
			let t2 = output.operations[r].t2;
			let t3 = output.operations[r].t3;
			let x = 0;
			for (let s = 0; s < input.item[i]; s++) {
				let m = output.operations[r].m[s];
				if (m == -1) {
					continue;
				}
				if (input.boms[i][m] == undefined) {
					continue;
				}
				x += input.boms[i][m].x;
			}
			this.grossP += pr;
			let delay = Math.max(0, t3 - d);
			let pe1 = Math.ceil(pr * delay / a);
			let pe2 = (t2 - t1) * x;
			this.dcost += pe1;
			this.scost += pe2;
		}
		this.profit = this.grossP - this.dcost - this.scost;
	}
}

class Tester {
	constructor(inputContent, outputContent) {
		let input = new InputFile(inputContent);
		let output = new OutputFile(outputContent, input);
		this.frame = new TesterFrame(input, output);
	}
}

class Visualizer {
	constructor() {
		this.grossProfitInput = document.getElementById('grossP');
		this.dcostSumInput = document.getElementById('d-cost');
		this.scostSumInput = document.getElementById('s-cost');
		this.totalProfitInput = document.getElementById('score');
		this.tooltipDiv = document.getElementById("tooltip");
		this.drawingElement = document.getElementById('drawing');
		this.zoominButton = document.getElementById('zoomin');
		this.zoomoutButton = document.getElementById('zoomout');
		this.resizeButton = document.getElementById('resize');
		this.padding = 14;
		this.width = this.drawingElement.clientWidth - (2 * this.padding);
		this.height = this.drawingElement.clientHeight - (2 * this.padding);
		this.svg_text = [];
		this.selectedRect = undefined;
	}

	addStringText(string, x, y, anchor, fontSize, fontFamily, color, className, m) {
		if (className == undefined) {
			className = "";
		}
		else {
			className = "\" class=\"" + className;
		}
		this.svg_text.push({ text: "<text x=\"" + x + "\" y=\"" + y + "\" text-anchor=\"" + anchor + "\" font-size=\"" + fontSize + "\" font-family=\"" + fontFamily + "\" fill=\"" + color + className + "\">" + string + "</text>", m: m });
	}

	addLineText(x1, y1, x2, y2, width, color, r, m) {
		let str = "";
		if (r !== undefined) {
			str = "\" class=\"line" + r;
		}
		this.svg_text.push({ text: "<line x1=\"" + x1 + "\" y1=\"" + y1 + "\" x2=\"" + x2 + "\" y2=\"" + y2 + "\" stroke-width=\"" + width + "\" stroke=\"" + color + str + "\" />", m: m});
	}

	addRectText(x, y, width, height, color, strokeWidth, strokeColor, opacity, r, s, m) {
		this.svg_text.push({ text: "<rect x=\"" + x + "\" y=\"" + y + "\" width=\"" + width + "\" height=\"" + height + "\" fill=\"" + color + "\" stroke-width=\"" + strokeWidth + "\" stroke=\"" + strokeColor + "\" opacity=\"" + opacity + "\" class=\"rect" + r + "_" + s + "\"></rect>", m: m });
	}

	addPolygonText(string, color, className, m) {
		this.svg_text.push({ text: "<polygon points=\"" + string + "\" fill=\"" + color + "\" class=\"" + className + "\"/>", m: m });
	}

	orderToHTML(r, s, input, output) {
		let delay = Math.max(0, output.operations[r].t3 - input.orders[r].d);
		let pe1 = Math.ceil(input.orders[r].pr * delay / input.orders[r].a);
		let pe2 = output.operations[r].t2 - output.operations[r].t1;

		return "オーダ番号: " + r + "<br>" +
			"品目番号: " + input.orders[r].i + "<br>" +
			"資源種別番号:" + s + "<br>" +
			"製造数量: " + input.orders[r].q + "<br>" +
			"最早開始時刻: " + input.orders[r].e + "秒<br>" +
			"納期時刻: " + input.orders[r].d + "秒<br>" +
			"納期遅れ許容時間: " + input.orders[r].a + "秒<br>" +
			"段取り開始時刻: " + output.operations[r].t1 + "秒<br>" +
			"製造開始時刻: " + output.operations[r].t2 + "秒<br>" +
			"製造終了時刻: " + output.operations[r].t3 + "秒<br>" +
			"粗利金額: " + input.orders[r].pr + "<br>" +
			"納期遅れペナルティ額: " + pe1 + "<br>" +
			"段取り時間ペナルティ額: " + pe2 + "<br>" +
			"粗利−ペナルティ: " + (input.orders[r].pr - pe1 - pe2) + "<br>";
	}

	drawFacilityLine(input, output) {
		for (let m = 0; m < input.M; ++m) {
			let bx1 = this.verticalLabelWidth + this.verticalTextSize;
			let bx2 = this.width;
			let by = this.rowHeight - this.chartpadding;
			this.addStringText(m, bx1 / 2, this.rowHeight * 0.6, "middle", this.rowHeight * 0.2, 'Menlo, sans-serif', '#000', undefined, m);
			this.addLineText(bx1, by, bx2, by, 0.5, "#000", undefined, m);
			for (let t = 0; t <= output.max_t; t += this.verticalbarinterval) {
				let x = (this.chartWidth / output.max_t) * t + this.verticalLabelWidth + this.verticalTextSize;
				let y1 = this.machineDetailHeight + this.chartpadding;
				let y2 = this.rowHeight - this.chartpadding;
				this.addLineText(x, y1, x, y2, 0.5, "#000", undefined, m);
				if (output.max_t <= 30 * this.verticalbarinterval) {
					this.addStringText(t + "", x, y1 - this.verticalTextSize * 0.5 + 10, "end", this.verticalTextSize, 'Menlo, sans-serif', '#000', undefined, m);
				} else if (output.max_t / this.verticalbarinterval <= 99) {
					this.addStringText(t + "", x, y1 - this.verticalTextSize * 0.4, "end", this.verticalTextSize * 0.8, 'Menlo, sans-serif', '#000', undefined, m);
				} else {
					this.addStringText(t + "", x, y1 - this.verticalTextSize * 0.2, "end", this.verticalTextSize * 0.4, 'Menlo, sans-serif', '#000', undefined, m);
				}
			}
		}
	}

	drawOrderRect(input, output) {
		for (let r = 0; r < input.R; ++r) {
			let i = input.orders[r].i;
			let dandoriColor = '#548235';
			let itemColor = this.color(i);
			for (let j = 0; j < input.item[i]; j++) {
				let m = output.operations[r].m[j];
				let t1 = output.operations[r].t1;
				let t2 = output.operations[r].t2;
				let t3 = output.operations[r].t3;
				if (m == -1) {
					continue;
				}
				while (t2 > t1) {
					if (input.boms[i][m] == undefined || input.boms[i][m].x == 0) {
						break;
					}
					let nPeriod = parseInt((t1 + this.verticalbarinterval) / this.verticalbarinterval);
					nPeriod = Math.max(1, nPeriod);
					let height = (this.chartHeight - 2 * this.chartpadding) * 0.8;
					let width = Math.min(t2 - t1, nPeriod * this.verticalbarinterval - t1) * this.chartWidth / output.max_t;
					let x = this.verticalLabelWidth + this.verticalTextSize + t1 * this.chartWidth / output.max_t;
					let y = this.rowHeight - this.chartpadding - height;
					this.addRectText(x, y, width, height, dandoriColor, 1, "#000", 0.8, r, j, m);
					t1 = Math.min(t2, nPeriod * this.verticalbarinterval);
				}
				while (t3 > t2) {
					if (input.boms[i][m] == undefined || input.boms[i][m].y == 0) {
						break;
					}
					let nPeriod = parseInt((t2 + this.verticalbarinterval) / this.verticalbarinterval);
					nPeriod = Math.max(1, nPeriod);
					let height = (this.chartHeight - 2 * this.chartpadding);
					let width = Math.min(t3 - t2, nPeriod * this.verticalbarinterval - t2) * this.chartWidth / output.max_t;
					let x = this.verticalLabelWidth + this.verticalTextSize + t2 * this.chartWidth / output.max_t;
					let y = this.rowHeight - this.chartpadding - height;
					this.addRectText(x, y, width, height, itemColor, 1, "#000", 0.8, r, j, m);
					t2 = Math.min(t3, nPeriod * this.verticalbarinterval);
				}
			}
		}
	}

	drawDeadlineOver(input, output) {
		for (let r = 0; r < input.R; ++r) {
			if (input.orders[r].d < output.operations[r].t3) {
				let m = output.operations[r].m[0];
				let t1 = output.operations[r].t1;
				let t3 = output.operations[r].t3;
				let x = this.verticalLabelWidth + this.verticalTextSize + ((t1 + t3) / 2) * this.chartWidth / output.max_t;
				let y = this.rowHeight + this.chartpadding * 0.85 - this.chartHeight;
				this.addStringText("!", x - this.verticalTextSize / 2.1, y, "start", 1.8 * this.verticalTextSize, 'Menlo, sans-serif', '#f00', undefined, m);
			}
		}
	}

	drawStartline(input, output, r, m, triangleMargin, triangleWidth, triangleHeight, labelVerticalMargin) {
		let x = this.verticalLabelWidth + this.verticalTextSize + (this.chartWidth / output.max_t) * input.orders[r].e;
		let y1 = this.machineDetailHeight + this.chartpadding;
		let y2 = this.rowHeight - this.chartpadding;
		let triangle = [[x, y2 + triangleMargin], [x - triangleWidth / 2, y2 + triangleMargin + triangleHeight], [x + triangleWidth / 2, y2 + triangleMargin + triangleHeight]];
		let label = 'e=' + input.orders[r].e;
		this.addLineText(x, y1, x, y2, 0, "#00f", r, m);
		this.addPolygonText(triangle.map((arr) => arr.join(",")).join(" "), "none", "poly_e" + r, m);
		this.addStringText(label, x, y2 + labelVerticalMargin + 1, 'middle', this.machineDetailHeight * 0.9, 'Menlo, sans-serif', "none", "text" + r, m);
		return x + 40;
	}

	drawDeadline(input, output, r, m, triangleMargin, triangleWidth, triangleHeight, labelVerticalMargin, labelLeft) {	
		let x = this.verticalLabelWidth + this.verticalTextSize + (this.chartWidth / output.max_t) * input.orders[r].d;
		let y1 = this.machineDetailHeight + this.chartpadding;
		let y2 = this.rowHeight - this.chartpadding;
		let triangle = [[x, y2 + triangleMargin], [x - triangleWidth / 2, y2 + triangleMargin + triangleHeight], [x + triangleWidth / 2, y2 + triangleMargin + triangleHeight]];
		let label = 'd=' + input.orders[r].d;
		this.addLineText(x, y1, x, y2, 0, "#f80", r, m);
		this.addPolygonText(triangle.map((arr) => arr.join(",")).join(" "), "none", "poly_d" + r, m);
		this.addStringText(label, Math.max(x, labelLeft), y2 + labelVerticalMargin + 1, 'middle', this.machineDetailHeight * 0.9, 'Menlo, sans-serif', "none", "text" + r, m);
		return Math.max(x, labelLeft) + 50;
	}

	drawDelayLine(input, output, r, m, triangleMargin, triangleWidth, triangleHeight, labelVerticalMargin, labelLeft) {
		let x = this.verticalLabelWidth + this.verticalTextSize + (this.chartWidth / output.max_t) * (input.orders[r].d + input.orders[r].a);
		let y1 = this.machineDetailHeight + this.chartpadding;
		let y2 = this.rowHeight - this.chartpadding;
		let triangle = [[x, y2 + triangleMargin], [x - triangleWidth / 2, y2 + triangleMargin + triangleHeight], [x + triangleWidth / 2, y2 + triangleMargin + triangleHeight]];
		let label = 'd+a=' + (input.orders[r].d + input.orders[r].a);
		if (input.orders[r].a) {
			this.addLineText(x, y1, x, y2, 0, "#f00", r, m);
			this.addPolygonText(triangle.map((arr) => arr.join(",")).join(" "), "none", "poly_a" + r, m);
			this.addStringText(label, Math.max(labelLeft, x), y2 + labelVerticalMargin + 1, 'middle', this.machineDetailHeight * 0.9, 'Menlo, sans-serif', "none", "text" + r, m);
		}
	}

	drawMark(input, output, r, triangleMargin, triangleWidth, triangleHeight) {
		let i = input.orders[r].i;
		for (let s = 0; s < input.item[i]; s++) {
			let m = output.operations[r].m[s];
			if (m == -1) {
				continue;
			}
			let start = output.operations[r].t1;
			let end = output.operations[r].t3;
			if (input.boms[i][m].x == 0) {
				start = output.operations[r].t2;
			}
			else if (input.boms[i][m].y == 0) {
				end = output.operations[r].t2;
			}
			let x = this.verticalLabelWidth + this.verticalTextSize + this.chartWidth / output.max_t * ((start + end) / 2);
			let y2 = this.rowHeight - this.chartpadding;
			let triangle = [[x, y2 + triangleMargin], [x - triangleWidth / 2, y2 + triangleMargin + triangleHeight], [x + triangleWidth / 2, y2 + triangleMargin + triangleHeight]];
			this.addPolygonText(triangle.map((arr) => arr.join(",")).join(" "), "none", "mark" + r, output.operations[r].m[s]);
		}
	}

	drawResourceNumber(input) {
		for (let i = 0; i < input.I; i++) {
			input.boms[i].forEach((value, index) => {
				this.addStringText("s=" + value.s, (this.verticalLabelWidth + this.verticalTextSize) / 2, this.rowHeight * 0.4, "middle", this.rowHeight * 0.1, "Menlo, sans-serif", "none", "subresource" + i, index);
			});
		}
	}

	drawLines(input, output, r, m) {
		let triangleMargin = 4;
		let triangleWidth = 13;
		let triangleHeight = 12;
		let labelVerticalMargin = 20;
		let labelLeft = this.drawStartline(input, output, r, m, triangleMargin, triangleWidth, triangleHeight, labelVerticalMargin);
		labelLeft = this.drawDeadline(input, output, r, m, triangleMargin, triangleWidth, triangleHeight, labelVerticalMargin, labelLeft);
		this.drawDelayLine(input, output, r, m, triangleMargin, triangleWidth, triangleHeight, labelVerticalMargin, labelLeft);
		this.drawMark(input, output, r, triangleMargin, triangleWidth, triangleHeight);
	}

	drawChart(frame) {
		this.width = this.drawingElement.clientWidth - (2 * this.padding);
		this.height = this.drawingElement.clientHeight - (2 * this.padding);
		let input = frame.input;
		let output = frame.output;

		this.rowHeight = 130;
		this.chartHeight = this.rowHeight * 0.8;
		this.machineDetailHeight = this.rowHeight * 0.075;
		this.verticalLabelWidth = this.width * 0.04;
		this.chartpadding = this.chartHeight * 0.2;
		this.maxHeight = Math.max(this.height, this.chartHeight * input.M + this.chartpadding);
		this.verticalTextSize = this.verticalLabelWidth * 0.2;
		this.chartWidth = this.width - this.verticalLabelWidth - this.verticalTextSize * 2;
		this.verticalbarinterval = Math.round(Math.pow(10, Math.floor(Math.log10(output.max_t)) - 1));
		this.svg_text = [];
		this.svg_text.push({ text: "<svg id=\"SVG\" overflow=\"hidden\" width=\"" + this.width + "\" height=\"" + this.height + "\" viewbox=\"0 0 " + this.width + " " + this.height + "\">", m: -50 });	//mの値は適当に小さいものだったらなんでも

		if (Math.log10(output.max_t) % 1 > 0.3010) {
			this.verticalbarinterval = Math.round(Math.pow(10, Math.floor(Math.log10(output.max_t)) - 1)) * 5;
		}
		
		this.drawFacilityLine(input, output);
		this.drawResourceNumber(input);
		this.drawOrderRect(input, output);
		this.drawDeadlineOver(input, output);
		
		for (let r = 0; r < input.R; ++r) {
			if (!output.operations[r].valid) continue;

			let m = output.operations[r].m[0];
			this.drawLines(input, output, r, m);
			
		}

		for (let m = 0; m < input.M; m++) {
			this.svg_text.push({ text: "<g id=\"material" + m + "\" display=\"visible\" transform=\"translate(0," + m * this.chartHeight + ")\">", m: m - 0.1 });
			this.svg_text.push({ text: "</g>", m: m + 0.1 });
		}

		this.svg_text.push({ text: "</svg>", m: 1000 });	
		this.svg_text.sort((a, b) => { return a.m === b.m ? 0 : (a.m < b.m ? -1 : 1); });
		this.drawingElement.innerHTML = this.svg_text.map((s) => s.text).join("");		
	}

	setScore(frame) {
		this.grossProfitInput.value = String(frame.grossP).replace(/(\d)(?=(\d\d\d)+(?!\d))/g, '$1,');
		this.dcostSumInput.value = String(frame.dcost).replace(/(\d)(?=(\d\d\d)+(?!\d))/g, '$1,');
		this.scostSumInput.value = String(frame.scost).replace(/(\d)(?=(\d\d\d)+(?!\d))/g, '$1,');
		this.totalProfitInput.value = String(frame.profit).replace(/(\d)(?=(\d\d\d)+(?!\d))/g, '$1,');
	}

	setZoomAction() {
		let _this = this;
		let svg = document.getElementById("SVG");
		this.zoominButton.addEventListener('click', function () {
			const [x, y, width, height] = svg.getAttribute("viewBox").split(" ").map(str => parseFloat(str));
			svg.setAttribute("viewBox", [x, y, width * 0.95, height * 0.95].join(" "));
		});
		this.zoomoutButton.addEventListener('click', function () {
			const [x, y, width, height] = svg.getAttribute("viewBox").split(" ").map(str => parseFloat(str));
			svg.setAttribute("viewBox", [x, y, width * 1.05, height * 1.05].join(" "));
		});
		this.resizeButton.addEventListener('click', function () {
			svg.setAttribute("viewBox", "0 0 " + _this.width + " " + _this.height);
		});
		this.drawingElement.addEventListener('mousedown', function (evt) {
			let x = evt.clientX;
			let y = evt.clientY;
			const [vx, vy, width, height] = svg.getAttribute("viewBox").split(" ").map(str => parseFloat(str));
			let mmove = function (evt) {
				let nx = evt.clientX;
				let ny = evt.clientY;
				svg.setAttribute("viewBox", [Math.max(0, Math.min(_this.width - width, vx - (nx - x))), Math.max(0, Math.min(_this.maxHeight - height, vy - (ny - y))), width, height].join(" "));
			}
			let mup = function () {
				_this.drawingElement.removeEventListener('mousemove', mmove);
				_this.drawingElement.removeEventListener('mouseup', mup);
			}
			_this.drawingElement.addEventListener('mousemove', mmove);
			_this.drawingElement.addEventListener('mouseup', mup);
		});
		this.drawingElement.addEventListener("wheel", (evt) => {
			const [vx, vy, width, height] = svg.getAttribute("viewBox").split(" ").map(str => parseFloat(str));
			let x = vx + evt.deltaX, y = vy + evt.deltaY;
			let b = false;
			if (x <= 0) {
				x = 0;
			}
			else if (x >= _this.width - width) {
				x = _this.width - width;
			}
			else {
				b = true;
			}
			if (y <= 0) {
				y = 0;
			}
			else if (y >= _this.maxHeight - height) {
				y = _this.maxHeight - height;
			}
			else {
				b = true;
			}
			if (b && evt.preventDefault) {
				evt.preventDefault();
			}
			svg.setAttribute("viewBox", [x, y, width, height].join(" "));
		});
		this.tooltipDiv.addEventListener("mousedown", (evt) => {
			if (evt.preventDefault) {
				evt.preventDefault();
			}
			let x = evt.clientX;
			let y = evt.clientY;
			let regex = /\d+/;
			let arr = regex.exec(_this.tooltipDiv.style.right);
			let px = parseFloat(arr[0]);
			let py = parseFloat(regex.exec(_this.tooltipDiv.style.bottom)[0]);
			let mmove = function (evt) {
				let nx = evt.clientX;
				let ny = evt.clientY;
				_this.tooltipDiv.style.right = (px - nx + x) + "px";
				_this.tooltipDiv.style.bottom = (py - ny + y) + "px";
			}
			let mup = function () {
				_this.tooltipDiv.removeEventListener('mousemove', mmove);
				_this.tooltipDiv.removeEventListener('mouseup', mup);
			}
			_this.tooltipDiv.addEventListener('mousemove', mmove);
			_this.tooltipDiv.addEventListener('mouseup', mup);
		});
		this.zoominButton.classList.remove('disabled');
		this.zoomoutButton.classList.remove('disabled');
		this.resizeButton.classList.remove('disabled');
	}

	draw(frame) {
		this.selectedRect = undefined;
		this.tooltipDiv.style.display = "none";
		this.setScore(frame);
		this.drawChart(frame);
		this.setZoomAction();
	}

	register(frame) {
		let input = frame.input;
		let output = frame.output;
		let tooltipDiv = this.tooltipDiv;
		let parent = document.getElementById("SVG");
		let height = this.chartHeight;
		let materials = [];
		for (let m = 0; m < input.M; m++) {
			materials[m] = document.getElementById("material" + m);
		}
		function reset() {
			for (let m = 0; m < input.M; m++) {
				materials[m].setAttribute("display", "visible");
				materials[m].setAttribute("transform", "translate(0," + m * height + ")");
			}
		}
		for (let r = 0; r < input.R; ++r) {
			if (!output.operations[r].valid) continue;
			let i = input.orders[r].i;
			let lines = Array.from(parent.getElementsByClassName("line" + r));
			let poly_e = Array.from(parent.getElementsByClassName("poly_e" + r));
			let poly_d = Array.from(parent.getElementsByClassName("poly_d" + r));
			let poly_a = Array.from(parent.getElementsByClassName("poly_a" + r));
			let marks = Array.from(parent.getElementsByClassName("mark" + r));
			let texts = Array.from(parent.getElementsByClassName("text" + r));
			let numbers = Array.from(parent.getElementsByClassName("subresource" + i));
			let rects = [];
			for (let s = 0; s < input.item[i]; ++s) {
				rects[s] = Array.from(parent.getElementsByClassName("rect" + r + "_" + s));
			}
			let material1 = [];
			input.boms[i].forEach((v, m) => material1.push(m));
			material1.sort((a, b) => { return a == b ? 0 : (a < b ? -1 : 1); });
			let material2 = output.operations[r].m.filter((value) => { return value != -1; });
			material2.sort((a, b) => { return a == b ? 0 : (a < b ? -1 : 1); });
			
			for (let s = 0; s < rects.length; s++) {
				let rect = rects[s];
				let str = this.orderToHTML(r, s, input, output);
				let _this = this;
				rect.forEach((rct) => {
					rct.addEventListener("mouseenter", (evt) => {
						if (_this.selectedRect === undefined) {
							rects.forEach((rc) => {
								rc.forEach((e) => e.setAttribute("stroke-width", "2.5"));
							});
							lines.forEach((line) => line.setAttribute("stroke-width", "2.5"));
							poly_e.forEach((poly) => poly.setAttribute("fill", "#00f"));
							poly_d.forEach((poly) => poly.setAttribute("fill", "#f80"));
							poly_a.forEach((poly) => poly.setAttribute("fill", "#f00"));
							texts.forEach((text) => text.setAttribute("fill", "#000"));
							marks.forEach((mark) => mark.setAttribute("fill", "#0f0"));
							numbers.forEach((num) => num.setAttribute("fill", "#000"));
							tooltipDiv.innerHTML = str;
							tooltipDiv.style.display = "block";
						}
					});
					rct.addEventListener("mouseleave", () => {
						if (_this.selectedRect === undefined) {
							rects.forEach((rc) => {
								rc.forEach((e) => e.setAttribute("stroke-width", "1"));
							});
							lines.forEach((line) => line.setAttribute("stroke-width", "0"));
							poly_e.forEach((poly) => poly.setAttribute("fill", "none"));
							poly_d.forEach((poly) => poly.setAttribute("fill", "none"));
							poly_a.forEach((poly) => poly.setAttribute("fill", "none"));
							texts.forEach((text) => text.setAttribute("fill", "none"));
							marks.forEach((mark) => mark.setAttribute("fill", "none"));
							numbers.forEach((num) => num.setAttribute("fill", "none"));
							tooltipDiv.style.display = "none";
						}
					});
					rct.addEventListener("click", (evt) => {
						let srect = _this.selectedRect;
						if (_this.selectedRect !== undefined) {
							_this.selectedRect = undefined;
							srect.dispatchEvent(new MouseEvent("mouseleave"));
						}
						if (srect === undefined || srect.className.baseVal != rct.className.baseVal) {
							rct.dispatchEvent(new MouseEvent("mouseenter", { clientX: evt.clientX, clientY: evt.clientY }));
							_this.selectedRect = rct;
							if ($("input[name='mode']:checked").val() == "mode0") {
								_this.maxHeight = Math.max(_this.height, _this.chartHeight * input.M + _this.chartpadding);
								reset();
							}
							if ($("input[name='mode']:checked").val() == "mode1") {
								_this.maxHeight = Math.max(_this.height, _this.chartHeight * material1.length + _this.chartpadding);
								for (let m = 0, x = 0; m < input.M; m++) {
									if (material1[x] == m) {
										materials[m].setAttribute("display", "visible");
										materials[m].setAttribute("transform", "translate(0," + x * height + ")");
										x++;
									}
									else {
										materials[m].setAttribute("display", "none");
									}
								}
							}
							if ($("input[name='mode']:checked").val() == "mode2") {
								_this.maxHeight = Math.max(_this.height, _this.chartHeight * material2.length + _this.chartpadding);
								for (let m = 0, x = 0; m < input.M; m++) {
									if (material2[x] == m) {
										materials[m].setAttribute("display", "visible");
										materials[m].setAttribute("transform", "translate(0," + x * height + ")");
										x++;
									}
									else {
										materials[m].setAttribute("display", "none");
									}
								}
							}
						}
					});
				});
			}
		}
		this.reset = reset;
	}

	color(r) {
		let colors = [
			'#ff4500', '#f0e68c', '#66cdaa', '#4169e1', '#eed4d4',
			'#696969', '#c71585', '#ff8c00', '#7cfc00', '#000080',
			'#ffd2b3', '#800080', '#8b4513', '#bdb76b', '#bdb76b',
			'#00ffff', '#000000', '#7b68ee', '#d2691e', '#e6e6fa'
		];
		return colors[r % colors.length];
	}
}

class App {
	constructor() {

		let _this = this;
		this.visualizer = new Visualizer();
		this.loader = new FileSelector();
		this.loader.callback = function (inputContent, outputContent) {
			_this.tester = new Tester(inputContent, outputContent);
			_this.visualizer.draw(_this.tester.frame);
			_this.visualizer.register(_this.tester.frame);

			$('.ui.radio.checkbox').change(() => {
				let rect = _this.visualizer.selectedRect;
				if (rect !== undefined) {
					let tooltipDiv = _this.visualizer.tooltipDiv;
					let left = tooltipDiv.style.left;
					let top = tooltipDiv.style.top;
					_this.visualizer.selectedRect = undefined;
					rect.dispatchEvent(new MouseEvent("click"));
					tooltipDiv.style.left = left;
					tooltipDiv.style.top = top;
				}
				else if ($("input[name='mode']:checked").val() == "mode0") {
					_this.visualizer.reset();
				}
				_this.visualizer.resizeButton.dispatchEvent(new Event("click"));
			});
		};
	}
}

let app;

window.onload = function () {
	app = new App();
};

window.addEventListener("resize", () => {
	if (app !== undefined) {
		app.visualizer.width = app.visualizer.drawingElement.clientWidth - (2 * app.visualizer.padding);
		app.visualizer.height = app.visualizer.drawingElement.clientHeight - (2 * app.visualizer.padding);
		let svg = document.getElementById("SVG");
		if (svg !== null) {
			svg.setAttribute("width", app.visualizer.width);
			svg.setAttribute("height", app.visualizer.height);
			app.visualizer.resizeButton.dispatchEvent(new Event("click"));
		}
	}
});
