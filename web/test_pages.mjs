// Verify the *deployed* page, not the local build.
//
// This fetches armemu.js and both guest tarballs from the live GitHub Pages URL and
// puts them through the exact code path index.html uses -- `DecompressionStream`,
// the twenty-line tar walk, MEMFS -- then runs the guests. Everything except the DOM
// is the same, which is enough to catch what actually goes wrong with a static
// deployment: a missing asset, a Content-Encoding that double-decompresses the
// tarball, an armemu.js built from different sources than the page expects.
//
//   node web/test_pages.mjs [base-url]
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';

const base = (process.argv[2] || 'https://yomei-o.github.io/arm_emu_cpp/').replace(/\/?$/, '/');
console.log('base:', base);

const grab = async (rel) => {
  const res = await fetch(base + rel);
  if (!res.ok) throw new Error(`${rel}: HTTP ${res.status}`);
  // The page relies on the server *not* setting Content-Encoding on a .tar.gz: with
  // it, fetch decompresses transparently and DecompressionStream then chokes on
  // already-plain bytes. Worth asserting rather than discovering in a browser.
  return { res, enc: res.headers.get('content-encoding') };
};

// armemu.js has to come off disk for `require`, so it lands in a temp file first.
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'armemu-pages-'));
{
  const { res } = await grab('web/armemu.js');
  fs.writeFileSync(path.join(tmp, 'armemu.js'), Buffer.from(await res.arrayBuffer()));
}
const createArmEmu = createRequire(import.meta.url)(path.join(tmp, 'armemu.js'));

let out = '';
const dec = new TextDecoder();
globalThis.armemuOutput = (fd, bytes) => { out += dec.decode(bytes, { stream: true }); };
globalThis.armemuLog = () => {};
const M = await createArmEmu();

const mkdirp = (p) => {
  let acc = '';
  for (const part of p.split('/').filter(Boolean)) { acc += '/' + part; try { M.FS.mkdir(acc); } catch {} }
};
const write = (p, bytes) => { mkdirp(p.split('/').slice(0, -1).join('/')); M.FS.writeFile(p, bytes); };
const cstr = (s) => {
  const n = M.lengthBytesUTF8(s) + 1, p = M._malloc(n);
  M.stringToUTF8(s, p, n);
  return p;
};
function run(guestPath, args) {
  out = '';
  const pp = cstr([guestPath, ...args].join('\0') + '\0\0');
  const cp = cstr(guestPath), rp = cstr('');
  let rc;
  try { rc = M._emu_run(cp, pp, rp, 0); } finally { M._free(pp); M._free(cp); M._free(rp); }
  return { rc, out, insns: M._emu_instructions() };
}

// Byte for byte what index.html does.
const td = new TextDecoder();
async function fetchTar(rel) {
  const { res, enc } = await grab(rel);
  if (enc) throw new Error(`${rel} is served with Content-Encoding: ${enc}; ` +
                           'fetch has already decompressed it and DecompressionStream will fail');
  const buf = new Uint8Array(await new Response(
    res.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  let off = 0, files = 0;
  while (off + 512 <= buf.length) {
    const hdr = buf.subarray(off, off + 512);
    if (hdr[0] === 0) break;
    let name = td.decode(hdr.subarray(0, 100)).replace(/\0.*$/, '');
    const prefix = td.decode(hdr.subarray(345, 500)).replace(/\0.*$/, '');
    if (prefix) name = prefix + '/' + name;
    const size = parseInt(td.decode(hdr.subarray(124, 136)).replace(/[\0 ]/g, ''), 8) || 0;
    const type = String.fromCharCode(hdr[156]);
    off += 512;
    if (name.startsWith('./')) name = name.slice(2);
    if (type === '5') mkdirp(name);
    else if (type === '0' || type === '\0' || type === '') { write('/' + name, buf.subarray(off, off + size)); files++; }
    off += Math.ceil(size / 512) * 512;
  }
  return files;
}

let pass = 0, fail = 0;
const check = (name, want, got) => {
  if (want === got) { console.log(`ok   ${name}`); pass++; }
  else { console.log(`FAIL ${name}\n     want: ${JSON.stringify(want)}\n     got:  ${JSON.stringify(got)}`); fail++; }
};

// ---- the two freestanding ELFs, fetched raw the way the page does
for (const [suffix, label] of [['elf', 'arm'], ['telf', 'thumb']]) {
  const { res } = await grab(`web/hello.${suffix}`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  write(`/hello.${suffix}`, bytes);
  check(`pages: hello (${label})`, 'hello from arm\n', run(`/hello.${suffix}`, []).out);
}

// ---- busybox
const bbFiles = await fetchTar('web/linux-guests.tar.gz');
mkdirp('bin');
M.FS.writeFile('/bin/busybox', M.FS.readFile('/busybox'));
console.log(`     (linux-guests.tar.gz: ${bbFiles} file(s))`);
check('pages: busybox echo', 'hi there\n', run('/bin/busybox', ['echo', 'hi', 'there']).out);
check('pages: busybox uname', 'armv7l\n', run('/bin/busybox', ['uname', '-m']).out);
{
  // The digest the page's own default command produces, against node's.
  const bytes = M.FS.readFile('/bin/busybox');
  const want = createHash('sha256').update(Buffer.from(bytes)).digest('hex');
  const got = (run('/bin/busybox', ['sha256sum', '/bin/busybox']).out || '').split(' ')[0];
  check('pages: busybox sha256sum (against node)', want, got);
}

// ---- CPython, with the page's own default expression
const pyFiles = await fetchTar('web/cpython-armv7.tar.gz');
console.log(`     (cpython-armv7.tar.gz: ${pyFiles} file(s))`);
{
  const want = createHash('sha256').update(Buffer.from('arm_emu_cpp')).digest('hex');
  const r = run('/usr/bin/python3', ['-c',
    "import sys,platform,hashlib;print(sys.version.split()[0],platform.machine());" +
    "print(hashlib.sha256(b'arm_emu_cpp').hexdigest())"]);
  if (r.rc !== 0) console.log('     (emulator said: ' + M.UTF8ToString(M._emu_error()) + ')');
  check('pages: cpython (the page default)', `3.12.13 armv7l\n${want}\n`, r.out);
  console.log(`     (${r.insns.toLocaleString()} instructions)`);
}

fs.rmSync(tmp, { recursive: true, force: true });
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
