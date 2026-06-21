import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';

type Landmark = { x: number; y: number; z: number };

export type GlassesStyle =
  | 'ski' | 'pixel' | 'party'
  | 'g0' | 'g1' | 'g2' | 'g3' | 'g4' | 'g5' | 'g6' | 'g7' | 'g8' | 'g9'
  | 'g10' | 'g11' | 'g12' | 'g13' | 'g14' | 'g15' | 'g16' | 'g17' | 'g18' | 'g19'
  | 'g20' | 'g21' | 'g22' | 'g23' | 'g24';

export type HatStyle =
  | 'top-hat' | 'baseball-cap' | 'cowboy-hat' | 'fox-hat' | 'frog-hat'
  | 'graduation-cap' | 'headphones' | 'pirate-hat' | 'sombrero' | 'wizard-hat' | 'cat-ears';

export const HAT_URLS: Record<HatStyle, string> = {
  'top-hat': '/models/hats/top-hat.glb',
  'baseball-cap': '/models/hats/baseball-cap.glb',
  'cowboy-hat': '/models/hats/cowboy-hat.glb',
  'fox-hat': '/models/hats/fox-hat.glb',
  'frog-hat': '/models/hats/frog-hat.glb',
  'graduation-cap': '/models/hats/graduation-cap.glb',
  'headphones': '/models/hats/headphones.glb',
  'pirate-hat': '/models/hats/pirate-hat.glb',
  'sombrero': '/models/hats/sombrero.glb',
  'wizard-hat': '/models/hats/wizard-hat.glb',
  'cat-ears': '/models/hats/cat-ears.glb',
};

export type EarringStyle = 'diamond-studs' | 'hoop-earrings' | 'pearl-earrings';

export const EARRING_URLS: Record<EarringStyle, string> = {
  'diamond-studs': '/models/earrings/diamond-studs.glb',
  'hoop-earrings': '/models/earrings/hoop-earrings.glb',
  'pearl-earrings': '/models/earrings/pearl-earrings.glb',
};

export type NecklaceStyle = 'necklace' | 'pearl-necklace';

export const NECKLACE_URLS: Record<NecklaceStyle, string> = {
  'necklace': '/models/necklaces/necklace.glb',
  'pearl-necklace': '/models/necklaces/pearl-necklace.glb',
};

export type TieStyle = 'necktie' | 'bowtie';

export const TIE_URLS: Record<TieStyle, { obj?: string, mtl?: string, glb?: string, scale?: number, scaleY?: number, dy?: number }> = {
  necktie: { glb: '/models/ties/Necktie.glb?v=3', scale: 0.8, scaleY: 1.25 }, // Stretched vertically to make it longer
  bowtie: { obj: '/models/ties/Bowtie_01.obj?v=3', mtl: '/models/ties/Bowtie_01.mtl?v=3', scale: 1.2, dy: 0.1 },
};

export type MaskStyle = 'clown-mask' | 'fox-head' | 'anon-mask' | 'gas-mask';

export const MASK_URLS: Record<MaskStyle, string> = {
  'clown-mask': '/models/masks/clown-mask.glb',
  'fox-head': '/models/masks/fox-head.glb',
  'anon-mask': '/models/masks/anon-mask.glb',
  'gas-mask': '/models/masks/gas-mask.glb',
};

// Mask width in IPD units (face width ≈ 3 IPD; fox head covers full head ≈ 5 IPD).
const MASK_IPD_SCALE: Record<MaskStyle, number> = {
  'clown-mask': 3.2,
  'fox-head':   4.5,
  'anon-mask':  3.2,
  'gas-mask':   3.2,
};

// Vertical offset from face.bridge in IPD units (negative = down).
// face.bridge = between eyes; nose tip ≈ -1.5 IPD; chin ≈ -3 IPD.
const MASK_UP_OFFSET: Record<MaskStyle, number> = {
  'clown-mask': -0.4,
  'fox-head':   -0.3,
  'anon-mask':  -1.9,
  'gas-mask':   -0.6,
};

// Pre-rotation applied before bbox computation.
// Other masks face -Z in OBJ → Ry(π) brings them to +Z (toward camera).
// clown-mask faces -X in OBJ → Ry(π/2) brings it to +Z.
const MASK_PRE_ROTATION: Record<MaskStyle, [number, number, number]> = {
  'clown-mask': [0, Math.PI / 2, 0],
  'fox-head':   [0, Math.PI,     0],
  'anon-mask':  [0, Math.PI,     0],
  'gas-mask':   [0, Math.PI,     0],
};

const HAT_OFFSETS: Record<HatStyle, { dy: number; dz: number; scale?: number }> = {
  'top-hat': { dy: 0.15, dz: -0.05, scale: 1.05 },
  'baseball-cap': { dy: -0.12, dz: 0.05, scale: 1.30 },
  'cowboy-hat': { dy: -0.18, dz: -0.05, scale: 1.45 },
  'fox-hat': { dy: -0.08, dz: 0.00, scale: 1.05 },
  'frog-hat': { dy: -0.08, dz: 0.00, scale: 1.05 },
  'graduation-cap': { dy: -0.15, dz: -0.02, scale: 1.05 },
  'headphones': { dy: -0.95, dz: -0.05, scale: 1.05 },
  'pirate-hat': { dy: -0.16, dz: -0.02, scale: 1.05 },
  'sombrero': { dy: -0.18, dz: -0.02, scale: 1.50 },
  'wizard-hat': { dy: -0.14, dz: -0.05, scale: 1.35 },
  'cat-ears': { dy: -0.45, dz: -0.02, scale: 1.00 },
};

export const GLB_URLS: Record<GlassesStyle, string> = {
  ski: '/models/glasses/ski-goggles.glb',
  pixel: '/models/glasses/pixel-glasses.glb',
  party: '/models/glasses/party-glasses.glb',
  g0: '/models/glasses/glasses-classic.glb',
  g1: '/models/glasses/glasses-1.glb',
  g2: '/models/glasses/glasses-2.glb',
  g3: '/models/glasses/glasses-3.glb',
  g4: '/models/glasses/glasses-4.glb',
  g5: '/models/glasses/glasses-5.glb',
  g6: '/models/glasses/glasses-6.glb',
  g7: '/models/glasses/glasses-7.glb',
  g8: '/models/glasses/glasses-8.glb',
  g9: '/models/glasses/glasses-9.glb',
  g10: '/models/glasses/glasses-10.glb',
  g11: '/models/glasses/glasses-11.glb',
  g12: '/models/glasses/glasses-12.glb',
  g13: '/models/glasses/glasses-13.glb',
  g14: '/models/glasses/glasses-14.glb',
  g15: '/models/glasses/glasses-15.glb',
  g16: '/models/glasses/glasses-16.glb',
  g17: '/models/glasses/glasses-17.glb',
  g18: '/models/glasses/glasses-18.glb',
  g19: '/models/glasses/glasses-19.glb',
  g20: '/models/glasses/glasses-20.glb',
  g21: '/models/glasses/glasses-21.glb',
  g22: '/models/glasses/glasses-22.glb',
  g23: '/models/glasses/glasses-23.glb',
  g24: '/models/glasses/glasses-24.glb',
};

const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];

function toVec3(lm: Landmark, W: number, H: number): THREE.Vector3 {
  return new THREE.Vector3(
    (lm.x - 0.5) * W,
    -(lm.y - 0.5) * H,
    -lm.z * W * 0.5,
  );
}

function avg(landmarks: Landmark[], indices: number[], W: number, H: number): THREE.Vector3 {
  const v = new THREE.Vector3();
  for (const i of indices) v.add(toVec3(landmarks[i], W, H));
  return v.divideScalar(indices.length);
}

function faceFrame(landmarks: Landmark[], W: number, H: number) {
  const lEye = avg(landmarks, [33, 133], W, H);
  const rEye = avg(landmarks, [263, 362], W, H);
  const chin = toVec3(landmarks[152], W, H);
  const bridge = lEye.clone().add(rEye).multiplyScalar(0.5);
  const ipd = lEye.distanceTo(rEye);

  const right = rEye.clone().sub(lEye).normalize();
  const tempUp = bridge.clone().sub(chin).normalize();
  const forward = new THREE.Vector3().crossVectors(right, tempUp).normalize();
  const up = new THREE.Vector3().crossVectors(forward, right).normalize();

  const rotMat = new THREE.Matrix4().makeBasis(right, up, forward);
  const quat = new THREE.Quaternion().setFromRotationMatrix(rotMat);

  return { bridge, ipd, quat, up, right, forward };
}

export class AREngine {
  readonly renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.OrthographicCamera;

  private glassesMap: Record<GlassesStyle, THREE.Group>;
  private activeGlasses: THREE.Group;
  private glassesEnabled = false;

  private hatMap: Record<HatStyle, THREE.Group>;
  private activeHat: THREE.Group;
  private hatEnabled = false;
  private activeHatStyle: HatStyle = 'top-hat';

  private earringMap: Record<EarringStyle, { L: THREE.Group; R: THREE.Group }>;
  private activeEarringL: THREE.Group;
  private activeEarringR: THREE.Group;
  private earringsEnabled = false;

  private necklaceMap: Record<NecklaceStyle, THREE.Group>;
  private activeNecklace: THREE.Group;
  private necklaceEnabled = false;

  private tieMap: Record<TieStyle, THREE.Group>;
  private activeTie: THREE.Group;
  private tieEnabled = false;

  private maskMap: Record<MaskStyle, THREE.Group>;
  private activeMask: THREE.Group;
  private maskEnabled = false;
  private activeMaskStyle: MaskStyle = 'anon-mask';

  private occluder: THREE.Mesh;
  private neckOccluder: THREE.Mesh;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.sortObjects = true;
    this.renderer.autoClear = false;

    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-400, 400, 225, -225, -1000, 1000);
    this.camera.position.z = 500;

    const ambient = new THREE.AmbientLight(0xffffff, 0.65);
    const key = new THREE.DirectionalLight(0xfffbe8, 1.1);
    key.position.set(1, 2, 3);
    const fill = new THREE.DirectionalLight(0xd0e8ff, 0.35);
    fill.position.set(-1, -0.5, 1);
    this.scene.add(ambient, key, fill);

    this.occluder = new THREE.Mesh(
      new THREE.BufferGeometry(),
      new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: true, side: THREE.FrontSide }),
    );
    this.occluder.renderOrder = 0;
    this.occluder.visible = false;
    this.scene.add(this.occluder);

    this.glassesMap = Object.fromEntries(
      (Object.keys(GLB_URLS) as GlassesStyle[]).map(k => [k, new THREE.Group()])
    ) as Record<GlassesStyle, THREE.Group>;
    for (const g of Object.values(this.glassesMap)) { g.renderOrder = 1; g.visible = false; this.scene.add(g); }
    this.activeGlasses = this.glassesMap.ski;

    this.hatMap = Object.fromEntries(
      (Object.keys(HAT_URLS) as HatStyle[]).map(k => {
        const g = new THREE.Group(); g.renderOrder = 1; g.visible = false;
        this.scene.add(g);
        return [k, g];
      })
    ) as Record<HatStyle, THREE.Group>;
    this.activeHat = this.hatMap['top-hat'];

    this.earringMap = Object.fromEntries(
      (Object.keys(EARRING_URLS) as EarringStyle[]).map(k => {
        const L = new THREE.Group(); L.renderOrder = 1; L.visible = false;
        const R = new THREE.Group(); R.renderOrder = 1; R.visible = false;
        this.scene.add(L, R);
        return [k, { L, R }];
      })
    ) as Record<EarringStyle, { L: THREE.Group; R: THREE.Group }>;
    this.activeEarringL = this.earringMap['hoop-earrings'].L;
    this.activeEarringR = this.earringMap['hoop-earrings'].R;

    this.necklaceMap = Object.fromEntries(
      (Object.keys(NECKLACE_URLS) as NecklaceStyle[]).map(k => {
        const g = new THREE.Group(); g.renderOrder = 1; g.visible = false;
        this.scene.add(g);
        return [k, g];
      })
    ) as Record<NecklaceStyle, THREE.Group>;
    this.activeNecklace = this.necklaceMap['necklace'];

    this.tieMap = Object.fromEntries(
      (Object.keys(TIE_URLS) as TieStyle[]).map(k => {
        const g = new THREE.Group(); g.renderOrder = 1; g.visible = false;
        this.scene.add(g);
        return [k, g];
      })
    ) as Record<TieStyle, THREE.Group>;
    this.activeTie = this.tieMap['necktie'];

    this.neckOccluder = new THREE.Mesh(
      new THREE.CylinderGeometry(0.5, 0.5, 1.0, 16),
      new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: true })
    );
    this.neckOccluder.renderOrder = 0;
    this.neckOccluder.visible = false;
    this.scene.add(this.neckOccluder);

    this.maskMap = Object.fromEntries(
      (Object.keys(MASK_URLS) as MaskStyle[]).map(k => {
        const g = new THREE.Group(); g.renderOrder = 2; g.visible = false;
        this.scene.add(g);
        return [k, g];
      })
    ) as Record<MaskStyle, THREE.Group>;
    this.activeMask = this.maskMap['anon-mask'];

    this.loadGLBModels();
    this.loadHatModels();
    this.loadEarringModels();
    this.loadNecklaceModels();
    this.loadTieModels();
    this.loadMaskModels();
  }

  private loadGLBModels(): void {
    const loader = new GLTFLoader();
    const load = (style: GlassesStyle, url: string): void => {
      loader.load(url, (gltf) => {
        const model = gltf.scene;
        model.updateWorldMatrix(true, true);
        const box = new THREE.Box3().setFromObject(model);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        model.traverse(child => {
          if (child instanceof THREE.Mesh) {
            const wp = new THREE.Vector3(); child.getWorldPosition(wp);
            const dx = wp.x - center.x;
            if (Math.abs(dx) > size.x * 0.30) child.userData.side = dx > 0 ? 1 : -1;
          }
        });
        model.position.sub(center);
        model.scale.setScalar(2.6 / Math.max(size.x, 0.001));
        this.glassesMap[style].add(model);
      }, undefined, () => { });
    };
    for (const [style, url] of Object.entries(GLB_URLS) as [GlassesStyle, string][]) load(style, url);
  }

  private loadHatModels(): void {
    // --- Procedural sombrero (GLB export is broken: 25 unit-spheres at origin) ---
    (() => {
      const g = this.hatMap['sombrero'];
      const mkMat = (c: number) =>
        new THREE.MeshStandardMaterial({ color: c, roughness: 0.78 });
      const tan = 0xc4891a;
      const dark = 0x1a0900;
      // brim: bottom face at group y=0
      const brim = new THREE.Mesh(new THREE.CylinderGeometry(1.50, 1.50, 0.07, 64), mkMat(tan));
      brim.position.y = 0.035; brim.frustumCulled = false;
      // lower crown
      const crownLow = new THREE.Mesh(new THREE.CylinderGeometry(0.52, 0.80, 0.80, 32), mkMat(tan));
      crownLow.position.y = 0.47; crownLow.frustumCulled = false;
      // dome cap
      const crownTop = new THREE.Mesh(
        new THREE.SphereGeometry(0.52, 32, 16, 0, Math.PI * 2, 0, Math.PI * 0.55),
        mkMat(tan),
      );
      crownTop.position.y = 0.87; crownTop.frustumCulled = false;
      // decorative band
      const band = new THREE.Mesh(new THREE.TorusGeometry(0.70, 0.04, 8, 48), mkMat(dark));
      band.rotation.x = Math.PI / 2; band.position.y = 0.12; band.frustumCulled = false;
      g.add(brim, crownLow, crownTop, band);
    })();

    const loader = new GLTFLoader();
    const load = (style: HatStyle, url: string): void => {
      if (style === 'sombrero') return; // handled above

      loader.load(url, (gltf) => {
        const model = gltf.scene;
        model.updateWorldMatrix(true, true);
        const box = new THREE.Box3().setFromObject(model);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());

        const s = 3.0 / Math.max(size.x, size.z, 0.001);

        // Pirate-hat (and similar Y-symmetric models) have their head-opening
        // at Y=0 (center), with brim drooping below and crown above.
        // Use center pivot so the hat wraps the head rather than floating above it.
        const useCenterPivot = style === 'pirate-hat' || Math.abs(center.y) < size.y * 0.12;
        const pivotY = useCenterPivot ? -center.y * s : -(center.y - size.y / 2) * s;

        model.position.set(-center.x * s, pivotY, -center.z * s);
        model.scale.setScalar(s);

        model.traverse(child => {
          if (child instanceof THREE.Mesh) {
            child.frustumCulled = false;
            const mats = Array.isArray(child.material) ? child.material : [child.material];
            mats.forEach(m => { if (m) { m.side = THREE.DoubleSide; m.needsUpdate = true; } });
          }
        });

        this.hatMap[style].add(model);
      }, undefined, (err) => console.error(`[AR] Hat load error [${style}]:`, err));
    };
    for (const [style, url] of Object.entries(HAT_URLS) as [HatStyle, string][]) load(style, url);
  }

  private loadEarringModels(): void {
    const loader = new GLTFLoader();
    const load = (style: EarringStyle, url: string): void => {
      loader.load(url, (gltf) => {
        const model = gltf.scene;
        model.rotation.x = -Math.PI / 2;
        model.updateWorldMatrix(true, true);
        const box = new THREE.Box3().setFromObject(model);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        model.position.set(-center.x, -(center.y + size.y / 2), -center.z);
        model.scale.setScalar(0.20 / Math.max(size.y, 0.001));
        this.earringMap[style].L.add(model);
        const modelR = model.clone(true);
        modelR.scale.x *= -1;
        this.earringMap[style].R.add(modelR);
      }, undefined, () => { });
    };
    for (const [style, url] of Object.entries(EARRING_URLS) as [EarringStyle, string][]) load(style, url);
  }

  private loadNecklaceModels(): void {
    const loader = new GLTFLoader();
    for (const [style, url] of Object.entries(NECKLACE_URLS) as [NecklaceStyle, string][]) {
      loader.load(url, (gltf) => {
        const model = gltf.scene;
        model.traverse(child => { if (child instanceof THREE.Mesh && child.material) child.material.side = THREE.DoubleSide; });
        model.updateWorldMatrix(true, true);
        const box = new THREE.Box3().setFromObject(model);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        model.position.set(-center.x, -(center.y + size.y / 2), -center.z);
        const wrapper = new THREE.Group();
        wrapper.add(model);
        wrapper.scale.setScalar(1.0 / Math.max(size.x, 0.001));
        this.necklaceMap[style].add(wrapper);
      });
    }
  }

  private loadTieModels(): void {
    const gltf = new GLTFLoader();
    const obj = new OBJLoader();
    const mtl = new MTLLoader();
    for (const [style, config] of Object.entries(TIE_URLS) as [TieStyle, any][]) {
      const process = (model: THREE.Object3D) => {
        model.traverse(c => {
          if (c instanceof THREE.Mesh) {
            if (Array.isArray(c.material)) c.material.forEach(m => m.side = THREE.DoubleSide);
            else if (c.material) c.material.side = THREE.DoubleSide;
          }
        });
        model.updateWorldMatrix(true, true);
        const b = new THREE.Box3().setFromObject(model);
        const s = b.getSize(new THREE.Vector3());
        const c = b.getCenter(new THREE.Vector3());
        // Pivot at the top-center-back: -c.z + s.z / 2 places the wrapper's origin at the back face of the bounding box!
        model.position.set(-c.x, -(c.y + s.y / 2) + (config.dy || 0) * s.y, -c.z + s.z / 2);
        const wrapper = new THREE.Group();
        wrapper.add(model);
        const baseScale = (config.scale || 0.8) / Math.max(s.x, 0.001);
        wrapper.scale.set(baseScale, (config.scaleY || config.scale || 0.8) / Math.max(s.x, 0.001), baseScale);
        this.tieMap[style].add(wrapper);
      };
      if (config.glb) gltf.load(config.glb, res => process(res.scene));
      else if (config.obj) mtl.load(config.mtl, m => { m.preload(); obj.setMaterials(m); obj.load(config.obj, process); });
    }
  }

  private loadMaskModels(): void {
    const loader = new GLTFLoader();
    for (const [style, url] of Object.entries(MASK_URLS) as [MaskStyle, string][]) {
      loader.load(url, (gltf) => {
        const model = gltf.scene;
        // Apply axis correction BEFORE bbox so scaling is computed in corrected space.
        // obj2gltf preserves the OBJ axes as-is; most face-mask OBJs are Z-up with the
        // face pointing toward +Y, so -90° around X brings the face to face +Z (camera).
        const [rx, ry, rz] = MASK_PRE_ROTATION[style];
        model.rotation.set(rx, ry, rz);
        model.updateWorldMatrix(true, true);
        const box = new THREE.Box3().setFromObject(model);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        model.position.set(-center.x, -center.y, -center.z);
        model.scale.setScalar(MASK_IPD_SCALE[style] / Math.max(size.x, size.y, 0.001));
        model.traverse(child => {
          if (child instanceof THREE.Mesh) {
            child.frustumCulled = false;
            const mats = Array.isArray(child.material) ? child.material : [child.material];
            mats.forEach(m => { if (m) { m.side = THREE.DoubleSide; m.needsUpdate = true; } });
          }
        });
        this.maskMap[style].add(model);
      }, undefined, (err) => console.error(`[AR] Mask load error [${style}]:`, err));
    }
  }

  private refreshOccluder(landmarks: Landmark[], W: number, H: number): void {
    const pts = FACE_OVAL.map(i => toVec3(landmarks[i], W, H));
    const center = new THREE.Vector3(); pts.forEach(p => center.add(p)); center.divideScalar(pts.length);
    center.z -= 8;
    const verts: number[] = [];
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length];
      verts.push(center.x, center.y, center.z, a.x, a.y, a.z, b.x, b.y, b.z);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    const old = this.occluder.geometry; this.occluder.geometry = geo; old.dispose();
  }

  update(landmarks: Landmark[], W: number, H: number, splitScreen = false): void {
    if (landmarks.length < 478) return;
    this.renderer.setSize(splitScreen ? W * 2 : W, H, false);
    this.camera.left = -W / 2; this.camera.right = W / 2;
    this.camera.top = H / 2; this.camera.bottom = -H / 2;
    this.camera.updateProjectionMatrix();

    if (splitScreen) { this.renderer.setViewport(0, 0, W, H); this.renderer.setScissor(0, 0, W, H); this.renderer.setScissorTest(true); }
    else { this.renderer.setViewport(0, 0, W, H); this.renderer.setScissorTest(false); }

    const face = faceFrame(landmarks, W, H);
    this.refreshOccluder(landmarks, W, H);

    if (this.activeGlasses.visible) {
      this.activeGlasses.position.copy(face.bridge).addScaledVector(face.forward, face.ipd * 0.14).addScaledVector(face.up, -face.ipd * 0.12);
      this.activeGlasses.quaternion.copy(face.quat);
      this.activeGlasses.scale.setScalar(face.ipd);
      const yaw = face.right.z;
      this.activeGlasses.traverse(c => { if ('side' in c.userData) c.visible = !(c.userData.side === 1 && yaw < -0.1) && !(c.userData.side === -1 && yaw > 0.1); });
    }

    if (this.activeHat.visible) {
      const config = HAT_OFFSETS[this.activeHatStyle] || { dy: -0.10, dz: 0 };
      const scaleVal = config.scale || 1.0;
      this.activeHat.position
        .copy(toVec3(landmarks[10], W, H))
        .addScaledVector(face.up, face.ipd * config.dy)
        .addScaledVector(face.forward, face.ipd * config.dz);
      this.activeHat.quaternion.copy(face.quat);
      this.activeHat.scale.setScalar(face.ipd * scaleVal);
    }

    if (this.activeEarringL.visible) {
      this.activeEarringL.position
        .copy(toVec3(landmarks[93], W, H))
        .addScaledVector(face.up, -face.ipd * 0.12)
        .addScaledVector(face.right, -face.ipd * 0.08)
        .addScaledVector(face.forward, -face.ipd * 0.12);
      this.activeEarringL.quaternion.copy(face.quat);
      this.activeEarringL.scale.setScalar(face.ipd);

      this.activeEarringR.position
        .copy(toVec3(landmarks[323], W, H))
        .addScaledVector(face.up, -face.ipd * 0.12)
        .addScaledVector(face.right, face.ipd * 0.08)
        .addScaledVector(face.forward, -face.ipd * 0.12);
      this.activeEarringR.quaternion.copy(face.quat);
      this.activeEarringR.scale.setScalar(face.ipd);
    }

    if (this.necklaceEnabled || this.tieEnabled) {
      if (this.necklaceEnabled) {
        const neckPos = toVec3(landmarks[152], W, H)
          .addScaledVector(face.up, -face.ipd * 0.38)
          .addScaledVector(face.forward, -face.ipd * 0.24);
        this.activeNecklace.position.copy(neckPos);
        this.activeNecklace.quaternion.copy(face.quat);
        this.activeNecklace.scale.set(face.ipd * 1.5, face.ipd * 1.5 * 0.75, face.ipd * 1.5);
      }
      if (this.tieEnabled) {
        const tiePos = toVec3(landmarks[152], W, H)
          .addScaledVector(face.up, -face.ipd * 0.35)
          .addScaledVector(face.forward, -face.ipd * 0.18);
        this.activeTie.position.copy(tiePos);
        this.activeTie.quaternion.copy(face.quat);
        this.activeTie.scale.setScalar(face.ipd * 0.75);
      }

      const occluderPos = toVec3(landmarks[152], W, H)
        .addScaledVector(face.up, -face.ipd * 0.60)
        .addScaledVector(face.forward, -face.ipd * 0.45);
      this.neckOccluder.position.copy(occluderPos);
      this.neckOccluder.quaternion.copy(face.quat);
      this.neckOccluder.scale.set(face.ipd * 1.3, face.ipd * 1.2, face.ipd * 0.8);
    }

    if (this.activeMask.visible) {
      this.activeMask.position
        .copy(face.bridge)
        .addScaledVector(face.forward, face.ipd * 0.05)
        .addScaledVector(face.up, face.ipd * MASK_UP_OFFSET[this.activeMaskStyle]);
      this.activeMask.quaternion.copy(face.quat);
      this.activeMask.scale.setScalar(face.ipd);
    }

    this.renderer.clear();
    this.renderer.render(this.scene, this.camera);
  }

  clear(): void { this.renderer.setScissorTest(false); this.renderer.clear(); }

  setAccessories(glasses: boolean, hat: boolean, earrings: boolean, necklace = false, tie = false, mask = false): void {
    this.glassesEnabled = glasses; this.activeGlasses.visible = glasses;
    this.hatEnabled = hat; this.activeHat.visible = hat;
    this.earringsEnabled = earrings; this.activeEarringL.visible = earrings; this.activeEarringR.visible = earrings;
    this.necklaceEnabled = necklace; this.activeNecklace.visible = necklace;
    this.tieEnabled = tie; this.activeTie.visible = tie;
    this.maskEnabled = mask; this.activeMask.visible = mask;
    this.occluder.visible = glasses || hat || earrings || necklace || tie;
    this.neckOccluder.visible = necklace || tie;
  }

  setMaskStyle(style: MaskStyle): void {
    this.activeMask.visible = false;
    this.activeMask = this.maskMap[style];
    this.activeMaskStyle = style;
    this.activeMask.visible = this.maskEnabled;
  }

  setMaskEnabled(enabled: boolean): void {
    this.maskEnabled = enabled;
    this.activeMask.visible = enabled;
  }

  setGlassesStyle(style: GlassesStyle): void { this.activeGlasses.visible = false; this.activeGlasses = this.glassesMap[style]; this.activeGlasses.visible = this.glassesEnabled; }
  setHatStyle(style: HatStyle): void { this.activeHat.visible = false; this.activeHat = this.hatMap[style]; this.activeHatStyle = style; this.activeHat.visible = this.hatEnabled; }
  setEarringStyle(style: EarringStyle): void { this.activeEarringL.visible = false; this.activeEarringR.visible = false; this.activeEarringL = this.earringMap[style].L; this.activeEarringR = this.earringMap[style].R; this.activeEarringL.visible = this.earringsEnabled; this.activeEarringR.visible = this.earringsEnabled; }
  setNecklaceStyle(style: NecklaceStyle): void { this.activeNecklace.visible = false; this.activeNecklace = this.necklaceMap[style]; if (this.necklaceEnabled) this.activeNecklace.visible = true; }
  setTieStyle(style: TieStyle): void { this.activeTie.visible = false; this.activeTie = this.tieMap[style]; if (this.tieEnabled) this.activeTie.visible = true; }

  dispose(): void {
    this.scene.traverse(obj => {
      if (obj instanceof THREE.Mesh) {
        obj.geometry.dispose();
        if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
        else if (obj.material) obj.material.dispose();
      }
    });
    this.renderer.dispose();
  }
}
