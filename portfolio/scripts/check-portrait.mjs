// Run: node portfolio/scripts/check-portrait.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
const bytes = readFileSync(new URL('../public/media/portrait.glb', import.meta.url));
assert.equal(bytes.toString('ascii', 0, 4), 'glTF');
assert.equal(bytes.readUInt32LE(4), 2);
assert.equal(bytes.readUInt32LE(8), bytes.length);
assert(bytes.length < 4_000_000, 'Keep the portrait below the 4 MB download budget');
const scene = JSON.parse(bytes.toString('utf8', 20, 20 + bytes.readUInt32LE(12)));
// These render in model-viewer without configuring a remote geometry decoder.
assert((scene.extensionsRequired ?? []).every(x => ['KHR_mesh_quantization', 'EXT_texture_webp'].includes(x)), 'Unexpected decoder requirement');
assert(scene.images.length > 0 && scene.images.every(x => x.bufferView !== undefined), 'Textures must be embedded');
assert(scene.buffers.every(x => !x.uri), 'Geometry must be embedded');
const primitive = scene.meshes[0].primitives[0];
assert(primitive.attributes.TEXCOORD_0 !== undefined, 'Portrait must retain its texture coordinates');
assert(scene.accessors[primitive.indices].count > 3000, 'Portrait must be a real mesh');
console.log('Portrait GLB: embedded textured mesh, supported extensions, download budget passed.');
