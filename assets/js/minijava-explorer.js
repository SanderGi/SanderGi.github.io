const root = document.querySelector("[data-minijava-demo]");

if (root) {
  const editorHost = root.querySelector("[data-minijava-editor]");
  const assemblyEl = root.querySelector("[data-minijava-assembly]");
  const compileButton = root.querySelector("[data-minijava-compile]");
  const targetSelect = root.querySelector("[data-minijava-target]");
  const statusEl = root.querySelector("[data-minijava-status]");

  const palette = [
    "rgba(244, 124, 84, 0.34)",
    "rgba(111, 207, 183, 0.38)",
    "rgba(30, 136, 229, 0.28)",
    "rgba(247, 196, 1, 0.36)",
    "rgba(177, 133, 219, 0.32)",
    "rgba(1, 191, 103, 0.28)",
    "rgba(255, 150, 73, 0.34)",
    "rgba(64, 207, 103, 0.28)",
  ];

  const starter = `class DispatchDemo {
  public static void main(String[] args) {
    System.out.println(new EchoChild().run(3));
  }
}

class EchoBase {
  int volume;
  public int run(int seed) {
    volume = seed;
    return this.speak(seed);
  }
  public int speak(int note) {
    return note + volume;
  }
}

class EchoChild extends EchoBase {
  boolean volume;
  int harmony;
  public int speak(int note) {
    harmony = note * 2;
    if (note < 2)
      harmony = harmony + 1;
    else
      harmony = harmony + this.sing(note - 1);
    return harmony;
  }
  public int sing(int note) {
    return note + 7;
  }
}`;

  let cm = null;
  let cmTools = null;
  let fallbackEditor = null;
  let traceRanges = [];
  let activeTrace = null;
  let markEffect = null;
  let activeMarkEffect = null;

  class Tokenizer {
    constructor(source) {
      this.source = source;
      this.tokens = [];
      this.index = 0;
      this.scan();
    }

    scan() {
      const keywords = new Set([
        "class",
        "public",
        "static",
        "void",
        "main",
        "String",
        "extends",
        "return",
        "int",
        "boolean",
        "if",
        "else",
        "while",
        "true",
        "false",
        "this",
        "new",
        "length",
        "System",
        "out",
        "println",
      ]);
      let i = 0;
      while (i < this.source.length) {
        const start = i;
        const ch = this.source[i];
        if (/\s/.test(ch)) {
          i += 1;
          continue;
        }
        if (ch === "/" && this.source[i + 1] === "/") {
          while (i < this.source.length && this.source[i] !== "\n") i += 1;
          continue;
        }
        if (ch === "/" && this.source[i + 1] === "*") {
          i += 2;
          while (i < this.source.length && !(this.source[i] === "*" && this.source[i + 1] === "/")) i += 1;
          i += 2;
          continue;
        }
        if (ch === "&" && this.source[i + 1] === "&") {
          this.tokens.push({ type: "&&", value: "&&", start, end: start + 2 });
          i += 2;
          continue;
        }
        if ("{}()[];,.=+-*!<".includes(ch)) {
          this.tokens.push({ type: ch, value: ch, start, end: start + 1 });
          i += 1;
          continue;
        }
        if (/[0-9]/.test(ch)) {
          i += 1;
          while (/[0-9]/.test(this.source[i] || "")) i += 1;
          this.tokens.push({ type: "number", value: this.source.slice(start, i), start, end: i });
          continue;
        }
        if (/[A-Za-z_]/.test(ch)) {
          i += 1;
          while (/[A-Za-z0-9_]/.test(this.source[i] || "")) i += 1;
          const value = this.source.slice(start, i);
          this.tokens.push({ type: keywords.has(value) ? value : "id", value, start, end: i });
          continue;
        }
        throw new Error(`Unexpected character '${ch}' at offset ${i}.`);
      }
      this.tokens.push({ type: "eof", value: "", start: this.source.length, end: this.source.length });
    }

    peek(offset = 0) {
      return this.tokens[this.index + offset] || this.tokens[this.tokens.length - 1];
    }

    match(type) {
      if (this.peek().type !== type) return null;
      return this.tokens[this.index++];
    }

    expect(type) {
      const token = this.match(type);
      if (!token) throw new Error(`Expected '${type}' near offset ${this.peek().start}.`);
      return token;
    }
  }

  class Parser {
    constructor(source) {
      this.tok = new Tokenizer(source);
    }

    parse() {
      const main = this.parseMainClass();
      const classes = [];
      while (this.tok.peek().type !== "eof") classes.push(this.parseClassDecl());
      return { type: "Program", main, classes, start: main.start, end: classes.at(-1)?.end || main.end };
    }

    parseMainClass() {
      const start = this.tok.expect("class").start;
      const name = this.identifier();
      this.tok.expect("{");
      const methodStart = this.tok.expect("public").start;
      this.tok.expect("static");
      this.tok.expect("void");
      const mainToken = this.tok.expect("main");
      this.tok.expect("(");
      this.tok.expect("String");
      this.tok.expect("[");
      this.tok.expect("]");
      const arg = this.identifier();
      this.tok.expect(")");
      this.tok.expect("{");
      const statement = this.parseStatement();
      this.tok.expect("}");
      const close = this.tok.expect("}");
      return { type: "MainClass", name, arg, statement, start, headerEnd: name.end, methodStart, methodHeaderEnd: mainToken.end, end: close.end };
    }

    parseClassDecl() {
      const start = this.tok.expect("class").start;
      const name = this.identifier();
      let parent = null;
      let extendsStart = null;
      if (this.tok.match("extends")) {
        extendsStart = this.tok.peek(-1).start;
        parent = this.identifier();
      }
      this.tok.expect("{");
      const fields = [];
      const methods = [];
      while (this.tok.peek().type !== "}") {
        if (this.tok.peek().type === "public") methods.push(this.parseMethodDecl());
        else fields.push(this.parseVarDecl());
      }
      const close = this.tok.expect("}");
      return { type: "ClassDecl", name, parent, fields, methods, start, headerEnd: name.end, extendsStart, end: close.end };
    }

    parseMethodDecl() {
      const start = this.tok.expect("public").start;
      const returnType = this.parseType();
      const name = this.identifier();
      this.tok.expect("(");
      const params = [];
      if (this.tok.peek().type !== ")") {
        params.push(this.parseFormal());
        while (this.tok.match(",")) params.push(this.parseFormal());
      }
      this.tok.expect(")");
      this.tok.expect("{");
      const locals = [];
      while (this.isTypeStart() && this.looksLikeVarDecl()) locals.push(this.parseVarDecl());
      const statements = [];
      while (this.tok.peek().type !== "return") statements.push(this.parseStatement());
      const returnToken = this.tok.expect("return");
      const result = this.parseExpression();
      const semi = this.tok.expect(";");
      const close = this.tok.expect("}");
      return { type: "MethodDecl", returnType, name, params, locals, statements, result, start, headerEnd: name.end, returnStart: returnToken.start, returnEnd: semi.end, end: close.end };
    }

    parseFormal() {
      const formalType = this.parseType();
      const name = this.identifier();
      return { type: "Formal", formalType, name, start: formalType.start, end: name.end };
    }

    parseVarDecl() {
      const varType = this.parseType();
      const name = this.identifier();
      const semi = this.tok.expect(";");
      return { type: "VarDecl", varType, name, start: varType.start, end: semi.end };
    }

    parseType() {
      const token = this.tok.peek();
      if (this.tok.match("int")) {
        if (this.tok.match("[")) {
          const close = this.tok.expect("]");
          return { type: "Type", name: "int[]", start: token.start, end: close.end };
        }
        return { type: "Type", name: "int", start: token.start, end: token.end };
      }
      if (this.tok.match("boolean")) return { type: "Type", name: "boolean", start: token.start, end: token.end };
      const id = this.identifier();
      return { type: "Type", name: id.name, start: id.start, end: id.end };
    }

    parseStatement() {
      const token = this.tok.peek();
      if (this.tok.match("{")) {
        const statements = [];
        while (this.tok.peek().type !== "}") statements.push(this.parseStatement());
        const close = this.tok.expect("}");
        return { type: "Block", statements, start: token.start, end: close.end };
      }
      if (this.tok.match("if")) {
        this.tok.expect("(");
        const test = this.parseExpression();
        this.tok.expect(")");
        const consequent = this.parseStatement();
        this.tok.expect("else");
        const alternate = this.parseStatement();
        return { type: "If", test, consequent, alternate, start: token.start, end: alternate.end };
      }
      if (this.tok.match("while")) {
        this.tok.expect("(");
        const test = this.parseExpression();
        this.tok.expect(")");
        const body = this.parseStatement();
        return { type: "While", test, body, start: token.start, end: body.end };
      }
      if (this.tok.match("System")) {
        this.tok.expect(".");
        this.tok.expect("out");
        this.tok.expect(".");
        this.tok.expect("println");
        this.tok.expect("(");
        const value = this.parseExpression();
        this.tok.expect(")");
        const semi = this.tok.expect(";");
        return { type: "Print", value, start: token.start, end: semi.end };
      }
      const name = this.identifier();
      if (this.tok.match("[")) {
        const index = this.parseExpression();
        this.tok.expect("]");
        this.tok.expect("=");
        const value = this.parseExpression();
        const semi = this.tok.expect(";");
        return { type: "ArrayAssign", name, index, value, start: name.start, end: semi.end };
      }
      this.tok.expect("=");
      const value = this.parseExpression();
      const semi = this.tok.expect(";");
      return { type: "Assign", name, value, start: name.start, end: semi.end };
    }

    parseExpression(min = 0) {
      let left = this.parsePrefix();
      const precedence = { "&&": 1, "<": 2, "+": 3, "-": 3, "*": 4 };
      while (precedence[this.tok.peek().type] >= min) {
        const op = this.tok.peek();
        const prec = precedence[op.type];
        this.tok.index += 1;
        const right = this.parseExpression(prec + 1);
        left = { type: "Binary", op: op.type, left, right, start: left.start, end: right.end };
      }
      return left;
    }

    parsePrefix() {
      const token = this.tok.peek();
      if (this.tok.match("!")) {
        const expr = this.parsePrefix();
        return { type: "Not", expr, start: token.start, end: expr.end };
      }
      return this.parsePostfix();
    }

    parsePostfix() {
      let expr = this.parsePrimary();
      while (true) {
        if (this.tok.match("[")) {
          const index = this.parseExpression();
          const close = this.tok.expect("]");
          expr = { type: "ArrayLookup", array: expr, index, start: expr.start, end: close.end };
          continue;
        }
        if (this.tok.match(".")) {
          if (this.tok.match("length")) {
            expr = { type: "ArrayLength", array: expr, start: expr.start, end: this.tok.peek(-1)?.end || expr.end };
            continue;
          }
          const method = this.identifier();
          this.tok.expect("(");
          const args = [];
          if (this.tok.peek().type !== ")") {
            args.push(this.parseExpression());
            while (this.tok.match(",")) args.push(this.parseExpression());
          }
          const close = this.tok.expect(")");
          expr = { type: "Call", object: expr, method, args, start: expr.start, end: close.end };
          continue;
        }
        return expr;
      }
    }

    parsePrimary() {
      const token = this.tok.peek();
      if (this.tok.match("number")) return { type: "Int", value: Number(token.value), start: token.start, end: token.end };
      if (this.tok.match("true")) return { type: "Bool", value: true, start: token.start, end: token.end };
      if (this.tok.match("false")) return { type: "Bool", value: false, start: token.start, end: token.end };
      if (this.tok.match("this")) return { type: "This", start: token.start, end: token.end };
      if (this.tok.match("new")) {
        const start = token.start;
        if (this.tok.match("int")) {
          this.tok.expect("[");
          const size = this.parseExpression();
          const close = this.tok.expect("]");
          return { type: "NewArray", size, start, end: close.end };
        }
        const className = this.identifier();
        this.tok.expect("(");
        const close = this.tok.expect(")");
        return { type: "NewObject", className, start, end: close.end };
      }
      if (this.tok.match("(")) {
        const expr = this.parseExpression();
        const close = this.tok.expect(")");
        return { ...expr, start: token.start, end: close.end };
      }
      return { type: "Identifier", name: this.identifier(), start: token.start, end: token.end };
    }

    identifier() {
      const token = this.tok.expect("id");
      return { type: "IdentifierName", name: token.value, start: token.start, end: token.end };
    }

    isTypeStart() {
      return ["int", "boolean", "id"].includes(this.tok.peek().type);
    }

    looksLikeVarDecl() {
      if (this.tok.peek().type === "int" && this.tok.peek(1).type === "[" && this.tok.peek(2).type === "]") {
        return this.tok.peek(3).type === "id" && this.tok.peek(4).type === ";";
      }
      return this.tok.peek(1).type === "id" && this.tok.peek(2).type === ";";
    }
  }

  class Codegen {
    constructor(target) {
      this.target = target;
      this.lines = [];
      this.traces = [];
      this.nextTrace = 1;
      this.nextLabel = 1;
      this.classMap = new Map();
      this.classLayouts = new Map();
      this.currentClass = null;
      this.currentMethod = null;
    }

    compile(program) {
      this.classMap = new Map(program.classes.map((cls) => [cls.name.name, cls]));
      this.prepareLayouts(program.classes);
      const mainClassTrace = this.traceRange(program.main, program.main.start, program.main.headerEnd);
      const mainMethodTrace = this.traceRange(program.main, program.main.methodStart, program.main.methodHeaderEnd);
      this.header(mainClassTrace);
      this.section(program.main, "main method");
      this.emitLabel(this.external("asm_main"), mainMethodTrace);
      this.emitPrologue(mainMethodTrace);
      this.emit(this.target === "arm64-mac" ? "  mov x19, #0xBADBAD" : "  movq $0xBADBAD, %rdi", null, "`this` is not set in the static main method");
      this.statement(program.main.statement);
      this.emitEpilogue(mainMethodTrace);
      program.classes.forEach((cls) => {
        this.emit("");
        this.classDecl(cls);
      });
      return { text: this.lines.map((line) => line.text).join("\n"), traces: this.traces, lines: this.lines };
    }

    prepareLayouts(classes) {
      for (const cls of classes) {
        const parentLayout = cls.parent ? this.classLayouts.get(cls.parent.name) : null;
        const methods = parentLayout ? parentLayout.methods.slice() : [];
        for (const method of cls.methods) {
          const entry = { owner: cls.name.name, method };
          const index = methods.findIndex((item) => item.method.name.name === method.name.name);
          if (index >= 0) methods[index] = entry;
          else methods.push(entry);
        }
        const fields = parentLayout ? parentLayout.fields.slice() : [];
        for (const field of cls.fields) fields.push({ owner: cls.name.name, field });
        const fieldOffsets = new Map(fields.map((item, index) => [item.field.name.name, (index + 1) * 8]));
        const methodOffsets = new Map(methods.map((item, index) => [item.method.name.name, (index + 1) * 8]));
        this.classLayouts.set(cls.name.name, { cls, parent: cls.parent?.name || null, methods, fields, fieldOffsets, methodOffsets });
      }
    }

    classDecl(cls) {
      this.currentClass = cls;
      const layout = this.classLayouts.get(cls.name.name);
      this.emit(".data");
      if (this.target !== "x86") this.emit(this.target === "arm64-mac" ? "  .align 8" : "  .align 8");
      const classTrace = this.traceRange(cls, cls.start, cls.headerEnd);
      const parentTrace = cls.parent && cls.extendsStart != null ? this.traceRange(cls, cls.extendsStart, cls.parent.end) : classTrace;
      this.emitLabel(`${cls.name.name}$$`, classTrace);
      this.emit(this.wordDirective(`${layout.parent ? `${layout.parent}$$` : "0"}`), parentTrace, layout.parent ? `parent vtable: ${layout.parent}` : "no parent class");
      for (const item of layout.methods) {
        this.emit(this.wordDirective(`${item.owner}$${item.method.name.name}`), null, `${cls.name.name}.${item.method.name.name} at vtable offset ${layout.methodOffsets.get(item.method.name.name)}`);
      }
      this.emit(".text");
      for (const method of cls.methods) this.method(cls, method);
      this.currentClass = null;
    }

    method(cls, method) {
      this.currentClass = cls;
      this.currentMethod = method;
      this.section(method, `${cls.name.name}.${method.name.name}`);
      const methodTrace = this.traceRange(method, method.start, method.headerEnd);
      this.emitLabel(`${cls.name.name}$${method.name.name}`, methodTrace);
      this.emitPrologue(methodTrace);
      this.emitLocalSetup(method);
      for (const stmt of method.statements) this.statement(stmt);
      this.expr(method.result);
      this.emitEpilogue(this.traceRange(method, method.returnStart, method.returnEnd));
      this.currentMethod = null;
    }

    header(trace = null) {
      if (this.target === "arm64-mac") {
        this.emit(".text");
        this.emit(".global _asm_main", trace);
      } else {
        this.emit(this.target === "x86-mac" ? ".global _asm_main" : ".global asm_main", trace);
      }
    }

    section(node, label) {
      this.emit(`\n${this.comment(label)}`);
    }

    external(name) {
      return this.target === "x86-mac" || this.target === "arm64-mac" ? `_${name}` : name;
    }

    emitPrologue(trace = null) {
      if (this.target === "arm64-mac") {
        this.emit("  stp x30, xzr, [sp, #-16]!", trace, "save link register");
        this.emit("  stp x29, xzr, [sp, #-16]!", trace, "save frame pointer");
        this.emit("  mov x29, sp", trace, "start stack frame");
      } else {
        this.emit("  pushq %rbp", trace, "prologue: save frame pointer");
        this.emit("  movq %rsp, %rbp", trace, "prologue: start stack frame");
      }
    }

    emitLocalSetup(method) {
      if (!method) return;
      const slots = [...method.params, ...method.locals];
      if (this.target === "arm64-mac") {
        const bytes = Math.ceil((slots.length * 8) / 16) * 16;
        if (bytes) this.emit(`  sub sp, sp, #${bytes}`, null, "reserve local slots");
        method.params.forEach((param, index) => this.emit(`  str x${index}, [x29, #-${this.localArmOffset(param.name.name)}]`, this.trace(param), `save parameter ${param.name.name}`));
        method.locals.forEach((local) => this.emit(`  str xzr, [x29, #-${this.localArmOffset(local.name.name)}]`, null, `zero local ${local.name.name}`));
      } else {
        if (slots.length) this.emit(`  sub $${slots.length * 8}, %rsp`, null, "reserve local slots");
        const argRegisters = ["%rsi", "%rdx", "%rcx", "%r8", "%r9"];
        method.params.forEach((param, index) => this.emit(`  movq ${argRegisters[index] || "%r9"}, ${this.localOffset(param.name.name)}(%rsp)`, this.trace(param), `save parameter ${param.name.name}`));
        method.locals.forEach((local) => this.emit(`  movq $0, ${this.localOffset(local.name.name)}(%rsp)`, null, `zero local ${local.name.name}`));
      }
    }

    emitEpilogue(trace = null) {
      if (this.target === "arm64-mac") {
        this.emit("  mov sp, x29", trace, "epilogue: discard locals");
        this.emit("  ldp x29, xzr, [sp], #16", trace, "restore frame pointer");
        this.emit("  ldp x30, xzr, [sp], #16", trace, "restore link register");
        this.emit("  ret", trace);
      } else {
        this.emit("  movq %rbp, %rsp", trace, "epilogue: discard locals");
        this.emit("  popq %rbp", trace, "restore frame pointer");
        this.emit("  ret", trace);
      }
    }

    statement(node) {
      if (node.type === "Block") return node.statements.forEach((stmt) => this.statement(stmt));
      const trace = this.trace(node);
      if (node.type === "Print") {
        this.expr(node.value);
        this.emitCall("put", trace, "print evaluated value");
      } else if (node.type === "Assign") {
        this.expr(node.value);
        this.emit(this.mov("result", `[${node.name.name}]`), trace, `assign ${node.name.name}`);
      } else if (node.type === "ArrayAssign") {
        this.expr(node.index);
        this.emit(this.target === "arm64-mac" ? "  mov x1, x0" : "  movq %rax, %rsi", trace, "array index");
        this.expr(node.value);
        this.emit(this.arrayWrite(), trace, `array write ${node.name.name}[index] = value`);
      } else if (node.type === "If") {
        const elseLabel = this.label("elseLabel");
        const doneLabel = this.label("endIfLabel");
        this.expr(node.test);
        this.emit(this.branchZero(elseLabel), trace, "if false jump");
        this.statement(node.consequent);
        this.emit(this.jump(doneLabel), trace);
        this.emitLabel(elseLabel);
        this.statement(node.alternate);
        this.emitLabel(doneLabel);
      } else if (node.type === "While") {
        const test = this.label("whileCondition");
        const loop = this.label("whileBody");
        this.emit(this.jump(test), trace, "check loop condition");
        this.emitLabel(loop);
        this.statement(node.body);
        this.emitLabel(test);
        this.expr(node.test);
        this.emit(this.branchNonZero(loop), trace, "repeat while true");
      }
    }

    expr(node) {
      const trace = this.trace(node);
      if (node.type === "Int") this.emit(this.loadImm(node.value), trace);
      else if (node.type === "Bool") this.emit(this.loadImm(node.value ? 1 : 0), trace);
      else if (node.type === "Identifier") this.emit(this.mov(`[${node.name.name}]`, "result"), trace);
      else if (node.type === "This") this.emit(this.mov("this", "result"), trace);
      else if (node.type === "NewObject") this.newObject(node, trace);
      else if (node.type === "NewArray") {
        this.expr(node.size);
        this.newArray(trace);
      } else if (node.type === "ArrayLength") {
        this.expr(node.array);
        this.emit(this.mov("[array.length]", "result"), trace);
      } else if (node.type === "ArrayLookup") {
        this.expr(node.array);
        this.emit(this.target === "arm64-mac" ? "  mov x3, x0" : "  pushq %rax", trace, "save array pointer");
        this.expr(node.index);
        this.emit(this.mov("[array + index]", "result"), trace);
      } else if (node.type === "Call") {
        this.expr(node.object);
        for (const arg of node.args) this.expr(arg);
        this.dynamicCall(node, trace);
      } else if (node.type === "Not") {
        this.expr(node.expr);
        this.emit(this.unaryNot(), trace);
      } else if (node.type === "Binary") {
        this.expr(node.left);
        this.emit(this.pushResult(), trace, "save left operand");
        this.expr(node.right);
        this.emit(this.binary(node.op), trace);
      }
    }

    trace(node) {
      if (!node.traceId) {
        node.traceId = this.traceRange(node, node.start, node.end);
      }
      return node.traceId;
    }

    traceRange(node, start, end) {
      const key = `${start}:${end}`;
      node.traceIds ||= {};
      if (!node.traceIds[key]) {
        const id = `mj-${this.nextTrace++}`;
        node.traceIds[key] = id;
        this.traces.push({ id, start, end, color: palette[(this.nextTrace - 2) % palette.length] });
      }
      return node.traceIds[key];
    }

    emit(text, trace = null, note = "") {
      this.lines.push({ text: note ? `${text}    ${this.comment(note)}` : text, trace });
    }

    emitLabel(label, trace = null) {
      this.lines.push({ text: `${label}:`, trace });
    }

    label(prefix) {
      return `.L_${prefix}_${this.nextLabel++}`;
    }

    loadImm(value) {
      if (this.target === "arm64-mac") return `  mov x0, #${value}`;
      return `  movq $${value}, %rax`;
    }

    mov(from, to) {
      if (this.target === "arm64-mac") {
        if (from === "result" && /^\[[A-Za-z_][A-Za-z0-9_]*\]$/.test(to)) return this.storeName(to.slice(1, -1));
        if (to === "result" && /^\[[A-Za-z_][A-Za-z0-9_]*\]$/.test(from)) return this.loadName(from.slice(1, -1));
        if (to === "result" && from === "this") return "  mov x0, x19";
        if (to === "result" && from === "[array.length]") return "  ldr x0, [x0]";
        if (to === "result" && from === "[array + index]") return "  ldr x0, [x3, x0, lsl #3]";
        return `  mov x0, x0    ${this.comment(`${to} <- ${from}`)}`;
      }
      if (from === "result" && /^\[[A-Za-z_][A-Za-z0-9_]*\]$/.test(to)) return this.storeName(to.slice(1, -1));
      if (to === "result" && /^\[[A-Za-z_][A-Za-z0-9_]*\]$/.test(from)) return this.loadName(from.slice(1, -1));
      if (to === "result" && from === "this") return "  movq %rdi, %rax";
      if (to === "result" && from === "[array.length]") return "  movq 0(%rax), %rax";
      if (to === "result" && from === "[array + index]") return "  popq %rdi\n  movq 8(%rdi,%rax,8), %rax";
      return `  movq %rax, %rax    ${this.comment(`${to} <- ${from}`)}`;
    }

    loadName(name) {
      if (this.isLocal(name)) {
        return this.target === "arm64-mac" ? `  ldr x0, [x29, #-${this.localArmOffset(name)}]` : `  movq ${this.localOffset(name)}(%rsp), %rax`;
      }
      return this.target === "arm64-mac" ? `  ldr x0, [x19, #${this.fieldOffset(name)}]` : `  movq ${this.fieldOffset(name)}(%rdi), %rax`;
    }

    storeName(name) {
      if (this.isLocal(name)) {
        return this.target === "arm64-mac" ? `  str x0, [x29, #-${this.localArmOffset(name)}]` : `  movq %rax, ${this.localOffset(name)}(%rsp)`;
      }
      return this.target === "arm64-mac" ? `  str x0, [x19, #${this.fieldOffset(name)}]` : `  movq %rax, ${this.fieldOffset(name)}(%rdi)`;
    }

    isLocal(name) {
      return !!this.currentMethod && [...this.currentMethod.params, ...this.currentMethod.locals].some((item) => item.name.name === name);
    }

    localOffset(name) {
      const slots = this.currentMethod ? [...this.currentMethod.params, ...this.currentMethod.locals] : [];
      const index = Math.max(0, slots.findIndex((item) => item.name.name === name));
      return index * 8;
    }

    localArmOffset(name) {
      return this.localOffset(name) + 8;
    }

    fieldOffset(name) {
      const layout = this.classLayouts.get(this.currentClass?.name.name || "");
      return layout?.fieldOffsets.get(name) || 8;
    }

    arrayWrite() {
      return this.target === "arm64-mac" ? "  str x0, [x3, x1, lsl #3]" : "  movq %rax, 8(%rdx,%rsi,8)";
    }

    pushResult() {
      return this.target === "arm64-mac" ? "  str x0, [sp, #-16]!" : "  pushq %rax";
    }

    binary(op) {
      if (this.target === "arm64-mac") {
        const map = { "+": "add x0, x1, x0", "-": "sub x0, x1, x0", "*": "mul x0, x1, x0", "<": "cmp x1, x0; cset x0, lt", "&&": "and x0, x1, x0" };
        return `  ldr x1, [sp], #16\n  ${map[op] || `// ${op}`}`;
      }
      const map = { "+": "addq %rcx, %rax", "-": "subq %rax, %rcx\n  movq %rcx, %rax", "*": "imulq %rcx, %rax", "<": "cmpq %rax, %rcx\n  setl %al\n  movzbq %al, %rax", "&&": "andq %rcx, %rax" };
      return `  popq %rcx\n  ${map[op] || `# ${op}`}`;
    }

    unaryNot() {
      return this.target === "arm64-mac" ? "  cmp x0, #0\n  cset x0, eq" : "  cmpq $0, %rax\n  sete %al\n  movzbq %al, %rax";
    }

    branchZero(label) {
      return this.target === "arm64-mac" ? `  cbz x0, ${label}` : `  cmpq $1, %rax\n  jne ${label}`;
    }

    branchNonZero(label) {
      return this.target === "arm64-mac" ? `  cbnz x0, ${label}` : `  cmpq $1, %rax\n  je ${label}`;
    }

    jump(label) {
      return this.target === "arm64-mac" ? `  b ${label}` : `  jmp ${label}`;
    }

    emitCall(name, trace, note) {
      const symbol = ["put", "mjcalloc", "check_bounds"].includes(name) ? this.external(name) : name;
      this.emit(this.target === "arm64-mac" ? `  bl ${symbol}` : `  call ${symbol}`, trace, note);
    }

    wordDirective(value) {
      return this.target === "arm64-mac" ? `  .8byte ${value}` : `  .quad ${value}`;
    }

    newObject(node, trace) {
      const layout = this.classLayouts.get(node.className.name);
      const size = ((layout?.fields.length || 0) + 1) * 8;
      if (this.target === "arm64-mac") {
        this.emit(`  mov x0, #${size}`, trace, `allocate ${node.className.name} object`);
        this.emitCall("mjcalloc", trace);
        this.emit(`  adrp x1, ${node.className.name}$$@PAGE`, trace, "load vtable page");
        this.emit(`  add x1, x1, ${node.className.name}$$@PAGEOFF`, trace, "load vtable address");
        this.emit("  str x1, [x0]", trace, "store vtable pointer");
      } else {
        this.emit(`  movq $${size}, %rdi`, trace, `allocate ${node.className.name} object`);
        this.emitCall("mjcalloc", trace);
        this.emit(`  leaq ${node.className.name}$$(%rip), %rsi`, trace, "load vtable address");
        this.emit("  movq %rsi, 0(%rax)", trace, "store vtable pointer");
      }
    }

    newArray(trace) {
      if (this.target === "arm64-mac") {
        this.emit("  mov x1, x0", trace, "remember requested length");
        this.emit("  add x0, x0, #1", trace, "include length slot");
        this.emit("  lsl x0, x0, #3", trace, "convert slots to bytes");
        this.emitCall("mjcalloc", trace, "allocate zeroed array");
        this.emit("  str x1, [x0]", trace, "store length");
      } else {
        this.emit("  pushq %rax", trace, "remember requested length");
        this.emit("  movq $0, %rdi", trace);
        this.emit("  leaq 8(%rdi,%rax,8), %rdi", trace, "convert length to bytes");
        this.emitCall("mjcalloc", trace, "allocate zeroed array");
        this.emit("  popq %rsi", trace, "restore length");
        this.emit("  movq %rsi, (%rax)", trace, "store length");
      }
    }

    dynamicCall(node, trace) {
      const offset = this.methodOffset(node.method.name);
      if (this.target === "arm64-mac") {
        this.emit("  mov x19, x0", trace, "callee object becomes `this`");
        this.emit("  ldr x16, [x19]", trace, "load vtable pointer");
        this.emit(`  ldr x16, [x16, #${offset}]`, trace, `method slot ${node.method.name}`);
        this.emit("  blr x16", trace, "dynamic dispatch");
      } else {
        this.emit("  movq %rax, %rdi", trace, "callee object becomes `this`");
        this.emit("  movq (%rdi), %rax", trace, "load vtable pointer");
        this.emit(`  leaq ${offset}(%rax), %rax`, trace, `method slot ${node.method.name}`);
        this.emit("  call *(%rax)", trace, "dynamic dispatch");
      }
    }

    methodOffset(name) {
      for (const layout of this.classLayouts.values()) {
        if (layout.methodOffsets.has(name)) return layout.methodOffsets.get(name);
      }
      return 8;
    }

    comment(text) {
      return this.target === "arm64-mac" ? `// ${text}` : `# ${text}`;
    }
  }

  async function setupEditor() {
    try {
      const cmBundle = await import("/assets/js/vendor/codemirror-minijava.bundle.js");
      cmTools = cmBundle;
      const tracePlugin = cmBundle.EditorView.domEventHandlers({
        mouseover(event) {
          const target = event.target.closest("[data-trace]");
          if (target) setActiveTrace(target.dataset.trace, "source", false);
        },
        click(event) {
          const target = event.target.closest("[data-trace]");
          if (target) {
            event.preventDefault();
            event.stopPropagation();
            setActiveTrace(target.dataset.trace, "source", true);
          }
        },
      });
      cm = new cmBundle.EditorView({
        parent: editorHost,
        doc: starter,
        extensions: [
          cmBundle.keymap.of(cmBundle.defaultKeymap),
          cmBundle.java(),
          cmBundle.syntaxHighlighting(cmBundle.defaultHighlightStyle),
          cmBundle.lineNumbers(),
          cmBundle.highlightActiveLineGutter(),
          cmBundle.highlightSpecialChars(),
          cmBundle.drawSelection(),
          cmBundle.EditorState.allowMultipleSelections.of(true),
          cmBundle.highlightActiveLine(),
          cmBundle.EditorView.lineWrapping,
          tracePlugin,
          cmBundle.EditorView.theme({
            "&": { height: "100%" },
            ".cm-content": { padding: "0.85em" },
          }),
        ],
      });
    } catch {
      fallbackEditor = document.createElement("textarea");
      fallbackEditor.className = "minijava-fallback-editor";
      fallbackEditor.spellcheck = false;
      fallbackEditor.value = starter;
      editorHost.append(fallbackEditor);
      setStatus("CodeMirror could not load, so this demo is using a plain editor.");
    }
  }

  function sourceText() {
    return cm ? cm.state.doc.toString() : fallbackEditor?.value || "";
  }

  function setStatus(message) {
    statusEl.textContent = message;
  }

  function escapeHtml(text) {
    return String(text)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function compile() {
    try {
      activeTrace = null;
      const output = compileSource(sourceText(), targetSelect.value);
      traceRanges = output.traces;
      renderAssembly(output.lines, output.traces);
      renderSourceMarks();
      setStatus(`Compiled ${output.traces.length} highlighted constructs.`);
    } catch (error) {
      assemblyEl.textContent = "";
      clearSourceMarks();
      setStatus(error.message);
    }
  }

  function compileSource(source, target = targetSelect.value) {
    const ast = new Parser(source).parse();
    return new Codegen(target).compile(ast);
  }

  function renderAssembly(lines, traces) {
    const traceMap = new Map(traces.map((trace) => [trace.id, trace]));
    assemblyEl.innerHTML = lines
      .map((line) => {
        const escaped = escapeHtml(line.text);
        if (!line.trace) return escaped;
        const trace = traceMap.get(line.trace);
        return `<span class="mj-asm" data-trace="${line.trace}" style="--mj-color:${trace.color}">${escaped}</span>`;
      })
      .join("\n");
  }

  function clearSourceMarks() {
    if (!cm || !markEffect) return;
    const effects = [markEffect.of(cmTools.Decoration.none)];
    if (activeMarkEffect) effects.push(activeMarkEffect.of(cmTools.Decoration.none));
    cm.dispatch({ effects });
  }

  function ensureSourceMarkFields() {
    if (!cm || !cmTools) return;
    const { Decoration, StateField, StateEffect, EditorView } = cmTools;
    if (!markEffect) {
      markEffect = StateEffect.define();
      activeMarkEffect = StateEffect.define();
      const markField = StateField.define({
        create: () => Decoration.none,
        update(value, tr) {
          value = value.map(tr.changes);
          for (const effect of tr.effects) if (effect.is(markEffect)) value = effect.value;
          return value;
        },
        provide: (field) => EditorView.decorations.from(field),
      });
      const activeMarkField = StateField.define({
        create: () => Decoration.none,
        update(value, tr) {
          value = value.map(tr.changes);
          for (const effect of tr.effects) if (effect.is(activeMarkEffect)) value = effect.value;
          return value;
        },
        provide: (field) => EditorView.decorations.from(field),
      });
      cm.dispatch({ effects: StateEffect.appendConfig.of([markField, activeMarkField]) });
    }
  }

  function renderSourceMarks() {
    if (!cm || !cmTools) return;
    ensureSourceMarkFields();
    const { Decoration } = cmTools;
    const ranges = traceRanges
      .slice()
      .sort((a, b) => a.start - b.start || b.end - a.end)
      .map((trace) =>
        Decoration.mark({
          class: "mj-mark",
          attributes: { "data-trace": trace.id, style: `--mj-color:${trace.color}` },
        }).range(trace.start, trace.end)
      );
    cm.dispatch({ effects: markEffect.of(Decoration.set(ranges, true)) });
    renderActiveSourceMark();
  }

  function renderActiveSourceMark() {
    if (!cm || !cmTools) return;
    ensureSourceMarkFields();
    const { Decoration } = cmTools;
    const ranges = activeTrace
      ? traceRanges
          .filter((trace) => trace.id === activeTrace)
          .map((trace) =>
            Decoration.mark({
              class: "mj-active-mark",
              attributes: { "data-trace": trace.id, style: `--mj-color:${trace.color}` },
            }).range(trace.start, trace.end)
          )
      : [];
    cm.dispatch({ effects: activeMarkEffect.of(Decoration.set(ranges, true)) });
  }

  function scrollTraceIntoView(id, origin) {
    if (!id) return;
    if (origin === "source") {
      const target = assemblyEl.querySelector(`[data-trace="${CSS.escape(id)}"]`);
      if (target) {
        const paneRect = assemblyEl.getBoundingClientRect();
        const targetRect = target.getBoundingClientRect();
        const targetTop = assemblyEl.scrollTop + targetRect.top - paneRect.top - assemblyEl.clientHeight / 2 + targetRect.height / 2;
        assemblyEl.scrollTo({ top: Math.max(0, targetTop), behavior: "smooth" });
      }
      return;
    }
    if (origin === "assembly" && cm && cmTools) {
      const trace = traceRanges.find((item) => item.id === id);
      if (trace) {
        const block = cm.lineBlockAt(trace.start);
        const targetTop = block.top - cm.scrollDOM.clientHeight / 2 + block.height / 2;
        cm.scrollDOM.scrollTo({ top: Math.max(0, targetTop), behavior: "smooth" });
      }
    }
  }

  function setActiveTrace(id, origin = null, shouldScroll = false) {
    activeTrace = id;
    root.querySelectorAll(".mj-asm").forEach((el) => el.classList.toggle("active", el.dataset.trace === id));
    renderActiveSourceMark();
    if (shouldScroll) scrollTraceIntoView(id, origin);
  }

  root.addEventListener("pointerover", (event) => {
    const target = event.target.closest("[data-trace]");
    if (target) setActiveTrace(target.dataset.trace, target.closest(".minijava-assembly") ? "assembly" : "source", false);
  });

  root.addEventListener("click", (event) => {
    const target = event.target.closest("[data-trace]");
    if (target) {
      event.preventDefault();
      event.stopPropagation();
      setActiveTrace(target.dataset.trace, target.closest(".minijava-assembly") ? "assembly" : "source", true);
    }
  });

  root.addEventListener("pointerout", (event) => {
    if (!event.relatedTarget || !root.contains(event.relatedTarget)) setActiveTrace(activeTrace);
  });

  compileButton.addEventListener("click", compile);
  targetSelect.addEventListener("change", compile);

  window.__minijavaExplorer = { compileSource };

  await setupEditor();
  compile();
}
