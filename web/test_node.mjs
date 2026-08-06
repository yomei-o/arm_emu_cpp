// Headless proof that the WebAssembly build runs the same guests the native one
// does. Node only for the harness -- the emulator is the same code, and the guest
// files go into MEMFS exactly as a page would put them there.
//
//   node web/test_node.mjs
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import path from 'node:path';

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, '..');
const createArmEmu = require(path.join(here, 'armemu.js'));

let out = '';
const dec = new TextDecoder();
globalThis.armemuOutput = (fd, bytes) => { out += dec.decode(bytes, { stream: true }); };
globalThis.armemuLog = () => {};

const M = await createArmEmu();

const mkdirp = (p) => {
  let acc = '';
  for (const part of p.split('/').filter(Boolean)) { acc += '/' + part; try { M.FS.mkdir(acc); } catch {} }
};

// Copy a host file into MEMFS, creating the directories on the way.
function put(guestPath, hostPath) {
  mkdirp(guestPath.split('/').slice(1, -1).join('/'));
  M.FS.writeFile(guestPath, new Uint8Array(fs.readFileSync(hostPath)));
}

function copyTree(hostRoot, rel) {
  for (const e of fs.readdirSync(path.join(hostRoot, rel), { withFileTypes: true })) {
    const r = rel ? rel + '/' + e.name : e.name;
    if (e.isDirectory()) { mkdirp(r); copyTree(hostRoot, r); }
    else if (e.isFile()) M.FS.writeFile('/' + r, new Uint8Array(fs.readFileSync(path.join(hostRoot, r))));
  }
}

const cstr = (s) => {
  const n = M.lengthBytesUTF8(s) + 1, p = M._malloc(n);
  M.stringToUTF8(s, p, n);
  return p;
};

function run(guestPath, args, guestRoot = '') {
  out = '';
  // argv is a NUL-separated, double-NUL-terminated block: no JSON parser in the C.
  const pp = cstr([guestPath, ...args].join('\0') + '\0\0');
  const cp = cstr(guestPath);
  const rp = cstr(guestRoot);
  const rc = M._emu_run(cp, pp, rp, 0);
  M._free(pp); M._free(cp); M._free(rp);
  return { rc, out, insns: M._emu_instructions() };
}

let pass = 0, fail = 0;
const check = (name, want, got) => {
  if (want === got) { console.log(`ok   ${name}`); pass++; }
  else { console.log(`FAIL ${name}\n     want: ${JSON.stringify(want)}\n     got:  ${JSON.stringify(got)}`); fail++; }
};

// ---- the freestanding tests, in both instruction sets, which need nothing but the
// ELF. The .telf is the same source built for Thumb, so this is the one place the
// browser build checks both decoders.
for (const [suffix, label] of [['elf', 'arm'], ['telf', 'thumb']]) {
  const p = path.join(root, `tests/hello.${suffix}`);
  if (!fs.existsSync(p)) { console.log(`skip wasm hello.${suffix} (run tests/run_tests.sh first)`); continue; }
  put(`/hello.${suffix}`, p);
  check(`wasm: hello (${label})`, 'hello from arm\n', run(`/hello.${suffix}`, []).out);
}

// ---- busybox, if it has been fetched
const bb = path.join(root, 'guests/busybox');
if (fs.existsSync(bb)) {
  put('/bin/busybox', bb);
  put('/README.md', path.join(root, 'README.md'));
  check('wasm: busybox echo', 'hi there\n', run('/bin/busybox', ['echo', 'hi', 'there']).out);
  check('wasm: busybox uname', 'armv7l\n', run('/bin/busybox', ['uname', '-m']).out);
  // The same cross-check the native suite makes: the guest's digest must equal the
  // host's digest of the same bytes.
  const { createHash } = await import('node:crypto');
  const want = createHash('sha256').update(fs.readFileSync(path.join(root, 'README.md'))).digest('hex');
  const got = (run('/bin/busybox', ['sha256sum', '/README.md']).out || '').split(' ')[0];
  check('wasm: busybox sha256sum', want, got);
} else {
  console.log('skip busybox (guests/busybox not present)');
}

// ---- CPython, if the sysroot has been fetched.
//
// Sixty megabytes goes into MEMFS, which is the honest cost of a dynamically linked
// guest with a standard library: the page has to supply every file the loader and
// the interpreter will open. A shipped demo would trim the stdlib rather than
// pretend this is free.
const pyRoot = path.join(root, 'guests/sysroot');
if (fs.existsSync(path.join(pyRoot, 'usr/bin/python3'))) {
  copyTree(pyRoot, '');
  const r = run('/usr/bin/python3',
                ['-c', 'import sys,platform;print(sys.version.split()[0],platform.machine())']);
  if (r.rc !== 0) console.log('     (emulator said: ' + M.UTF8ToString(M._emu_error()) + ')');
  check('wasm: cpython', '3.12.13 armv7l\n', r.out);

  // Threads inside WebAssembly, with no WebAssembly threads involved: the guest's
  // four pthreads are interleaved by the emulator's own scheduler on one wasm
  // instance, so this needs neither SharedArrayBuffer nor COOP/COEP headers -- which
  // static hosting like GitHub Pages cannot set anyway.
  const t = run('/usr/bin/python3', ['-c',
    'import threading\n' +
    'n=[0]; lock=threading.Lock()\n' +
    'def w():\n' +
    '    for _ in range(2000):\n' +
    '        with lock: n[0]+=1\n' +
    'ts=[threading.Thread(target=w) for _ in range(4)]\n' +
    '[t.start() for t in ts]; [t.join() for t in ts]\n' +
    'print(n[0])']);
  if (t.rc !== 0) console.log('     (emulator said: ' + M.UTF8ToString(M._emu_error()) + ')');
  check('wasm: cpython threading', '8000\n', t.out);
} else {
  console.log('skip cpython (guests/sysroot not present)');
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
