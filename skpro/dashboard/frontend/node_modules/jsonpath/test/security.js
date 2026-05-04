var assert = require('assert');
var jp = require('../');

suite('security', function() {

  var cleanup = function() {
    if (Object.prototype.polluted) {
      delete Object.prototype.polluted;
    }
  };

  teardown(function() {
    cleanup();
  });

  test('blocks prototype pollution via value()', function() {
    cleanup();
    var data = {};
    assert.throws(function() {
      jp.value(data, '$.__proto__.polluted', 'yes');
    }, /Unsafe key/);
    assert.equal(({}).polluted, undefined);
  });

  test('blocks prototype pollution via apply()', function() {
    cleanup();
    var data = { safe: { ok: true } };
    assert.throws(function() {
      jp.apply(data, '$.__proto__.polluted', function() { return 'yes'; });
    }, /Unsafe key/);
    assert.equal(({}).polluted, undefined);
  });

  test('blocks unsafe subscript access', function() {
    cleanup();
    var data = {};
    assert.throws(function() {
      jp.query(data, '$["__proto__"]["polluted"]');
    }, /Unsafe key/);
    assert.equal(({}).polluted, undefined);
  });

  test('blocks unsafe union access', function() {
    cleanup();
    var data = { safe: 1 };
    assert.throws(function() {
      jp.nodes(data, "$['safe','__proto__']");
    }, /Unsafe key/);
    assert.equal(({}).polluted, undefined);
  });

  suite('CVE-2026-1615: blocks code injection in filter/script expressions', function() {
    var data = { a: {}, b: [1, 2, 3] };

    test('rejects constructor access in filter expression', function() {
      assert.throws(function() {
        jp.query(data, '$[?(@.constructor)]');
      }, /Unsafe expression/);
    });

    test('rejects constructor.constructor in filter expression', function() {
      assert.throws(function() {
        jp.query(data, '$[?(@.constructor.constructor)]');
      }, /Unsafe expression/);
    });

    test('rejects chained constructor.constructor call: @.foo["constructor"]["constructor"](...)()', function() {
      assert.throws(function() {
        jp.query(data, '$[?(@.foo["constructor"]["constructor"]("return process")())]');
      }, /Unsafe expression/);
    });

    test('rejects __proto__ access in filter expression', function() {
      assert.throws(function() {
        jp.query(data, '$[?(@.__proto__)]');
      }, /Unsafe expression/);
    });

    test('rejects function call in filter expression', function() {
      assert.throws(function() {
        jp.query(data, '$[?(process.exit(1))]');
      }, /Unsafe expression/);
    });

    test('rejects constructor access in script expression', function() {
      var scriptData = { a: [1, 2, 3] };
      assert.throws(function() {
        jp.query(scriptData, '$[(@.constructor)]');
      }, /Unsafe expression/);
    });

    test('allows safe filter expressions', function() {
      var storeData = { store: { book: [ { price: 5 }, { price: 15 } ] } };
      var results = jp.query(storeData, '$..book[?(@.price<10)]');
      assert.deepEqual(results, [ { price: 5 } ]);
    });

    test('allows safe script expressions', function() {
      var bookData = { book: [ { id: 1 }, { id: 2 }, { id: 3 } ] };
      var results = jp.nodes(bookData, '$..book[(@.length-1)]');
      assert.deepEqual(results[0].value, { id: 3 });
    });

    test('rejects bracket notation constructor: @["constructor"]', function() {
      assert.throws(function() { jp.query(data, '$[?(@["constructor"])]'); }, /Unsafe expression/);
    });

    test('rejects bracket notation __proto__: @["__proto__"]', function() {
      assert.throws(function() { jp.query(data, '$[?(@["__proto__"])]'); }, /Unsafe expression/);
    });

    test('rejects bracket notation prototype: @["prototype"]', function() {
      assert.throws(function() { jp.query(data, '$[?(@["prototype"])]'); }, /Unsafe expression/);
    });

    test('rejects ObjectExpression with unsafe key: { "__proto__": @ }', function() {
      assert.throws(function() { jp.query(data, '$[?({ "__proto__": @ })]'); }, /Unsafe expression|Unexpected token/);
    });

    test('rejects ObjectExpression with unsafe key: { "constructor": @ }', function() {
      assert.throws(function() { jp.query(data, '$[?({ "constructor": @ })]'); }, /Unsafe expression|Unexpected token/);
    });

    test('rejects ObjectExpression with unsafe key: { "prototype": @ }', function() {
      assert.throws(function() { jp.query(data, '$[?({ "prototype": @ })]'); }, /Unsafe expression|Unexpected token/);
    });

    test('rejects unicode escape constructor in bracket: @["\\u0063onstructor"]', function() {
      assert.throws(function() { jp.query(data, '$[?(@["\\u0063onstructor"])]'); }, /Unsafe expression/);
    });

    test('rejects unicode escape __proto__ in bracket', function() {
      var path = '$[?(@["\\u005f\\u005fproto\\u005f\\u005f"])]';
      assert.throws(function() { jp.query(data, path); }, /Unsafe expression/);
    });

    test('rejects IIFE: (function(){return 1})()', function() {
      assert.throws(function() { jp.query(data, '$[?((function(){return 1})())]'); }, /Unsafe expression/);
    });

    test('rejects direct function call: process.exit(1)', function() {
      assert.throws(function() { jp.query(data, '$[?(process.exit(1))]'); }, /Unsafe expression/);
    });

    test('rejects require() call', function() {
      assert.throws(function() { jp.query(data, '$[?(require("fs"))]'); }, /Unsafe expression/);
    });

    test('rejects eval() call', function() {
      assert.throws(function() { jp.query(data, '$[?(eval("1"))]'); }, /Unsafe expression/);
    });

    test('rejects globalThis / global identifier', function() {
      assert.throws(function() { jp.query(data, '$[?(globalThis)]'); }, /Unsafe expression/);
      assert.throws(function() { jp.query(data, '$[?(global)]'); }, /Unsafe expression/);
    });

    test('rejects NewExpression: new Function("return 1")()', function() {
      assert.throws(function() { jp.query(data, '$[?(new Function("return 1")())]'); }, /Unsafe expression/);
    });

    test('rejects JSFuck-style: [] ["filter"]["constructor"]', function() {
      assert.throws(function() { jp.query(data, '$[?([]["filter"]["constructor"])]'); }, /Unsafe expression/);
    });

    test('rejects JSFuck-style constructor call (no @)', function() {
      assert.throws(function() { jp.query(data, '$[?([]["filter"]["constructor"]("return 1")())]'); }, /Unsafe expression/);
    });

    test('rejects sequence expression: (1, process.exit)(1)', function() {
      assert.throws(function() { jp.query(data, '$[?((1, process.exit)(1))]'); }, /Unsafe expression/);
    });

    test('rejects method call on @: @.valueOf()', function() {
      assert.throws(function() { jp.query(data, '$[?(@.valueOf())]'); }, /Unsafe expression/);
    });

    test('rejects method call: @.toString()', function() {
      assert.throws(function() { jp.query(data, '$[?(@.toString())]'); }, /Unsafe expression/);
    });

    test('rejects template literal in computed: @[`constructor`]', function() {
      assert.throws(function() { jp.query(data, '$[?(@[`constructor`])]'); }, /Unsafe expression|Unexpected token|ILLEGAL/);
    });

    test('rejects tagged template (code execution vector)', function() {
      assert.throws(function() { jp.query(data, '$[?(String.raw`x`)]'); }, /Unsafe expression|Unexpected token|ILLEGAL/);
    });

    test('rejects ArrowFunctionExpression', function() {
      assert.throws(function() { jp.query(data, '$[?((()=>1)())]'); }, /Unsafe expression|Unexpected token/);
    });

    test('rejects ThisExpression (this)', function() {
      assert.throws(function() { jp.query(data, '$[?(this)]'); }, /Unsafe expression/);
    });

    test('rejects script expression with constructor', function() {
      assert.throws(function() { jp.query(data, '$[(@.constructor)]'); }, /Unsafe expression/);
    });

    test('rejects script expression with call', function() {
      assert.throws(function() { jp.query(data, '$[((function(){return 0})())]'); }, /Unsafe expression/);
    });

    test('allows @.length (no call)', function() {
      var r = jp.query(data, '$[?(@.length)]');
      assert.ok(Array.isArray(r));
    });

    test('allows bracket with safe key: @["length"]', function() {
      var r = jp.query(data, '$[?(@["length"])]');
      assert.ok(Array.isArray(r));
    });

    test('allows @["@class"] (existing test pattern)', function() {
      var d = { DIV: [{ '@class': 'value', val: 5 }] };
      var r = jp.query(d, '$..DIV[?(@["@class"]=="value")]');
      assert.deepEqual(r, d.DIV);
    });

    test('rejects prototype access in filter', function() {
      assert.throws(function() { jp.query(data, '$[?(@.prototype)]'); }, /Unsafe expression/);
    });

    test('rejects comma/sequence that could hide call', function() {
      assert.throws(function() { jp.query(data, '$[?((0, eval)("1"))]'); }, /Unsafe expression/);
    });

    test('rejects AssignmentExpression', function() {
      assert.throws(function() { jp.query(data, '$[?((x=1)==1)]'); }, /Unsafe expression/);
    });

    test('rejects UpdateExpression (++, --)', function() {
      assert.throws(function() { jp.query(data, '$[?(@.x++)]'); }, /Unsafe expression/);
    });
  });
});
