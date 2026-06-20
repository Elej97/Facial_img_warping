import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

type Landmark = { x: number; y: number; z: number };

export type GlassesStyle =
  | 'ski' | 'pixel' | 'party'
  | 'g0' | 'g1' | 'g2' | 'g3' | 'g4' | 'g5' | 'g6' | 'g7' | 'g8' | 'g9'
  | 'g10' | 'g11' | 'g12' | 'g13' | 'g14' | 'g15' | 'g16' | 'g17' | 'g18' | 'g19'
  | 'g20' | 'g21' | 'g22' | 'g23' | 'g24';

export type HatStyle =
  | 'top-hat' | 'baseball-cap' | 'cowboy-hat' | 'fox-hat' | 'frog-hat'
  | 'graduation-cap' | 'headphones' | 'pirate-hat' | 'sombrero' | 'wizard-hat' | 'cat-ears';

const HAT_URLS: Record<HatStyle, string> = {
  'top-hat':        '/models/hats/top-hat.glb',
  'baseball-cap':   '/models/hats/baseball-cap.glb',
  'cowboy-hat':     '/models/hats/cowboy-hat.glb',
  'fox-hat':        '/models/hats/fox-hat.glb',
  'frog-hat':       '/models/hats/frog-hat.glb',
  'graduation-cap': '/models/hats/graduation-cap.glb',
  'headphones':     '/models/hats/headphones.glb',
  'pirate-hat':     '/models/hats/pirate-hat.glb',
  'sombrero':       '/models/hats/sombrero.glb',
  'wizard-hat':     '/models/hats/wizard-hat.glb',
  'cat-ears':       '/models/hats/cat-ears.glb',
};

export type EarringStyle = 'diamond-studs' | 'hoop-earrings' | 'pearl-earrings';

const EARRING_URLS: Record<EarringStyle, string> = {
  'diamond-studs':  '/models/earrings/diamond-studs.glb',
  'hoop-earrings':  '/models/earrings/hoop-earrings.glb',
  'pearl-earrings': '/models/earrings/pearl-earrings.glb',
};

export type NecklaceStyle = 'necklace' | 'pearl-necklace';

const NECKLACE_URLS: Record<NecklaceStyle, string> = {
  'necklace':       '/models/necklaces/necklace.glb',
  'pearl-necklace': '/models/necklaces/pearl-necklace.glb',
};

const GLB_URLS: Record<GlassesStyle, string> = {
  ski:   '/models/glasses/ski-goggles.glb',
  pixel: '/models/glasses/pixel-glasses.glb',
  party: '/models/glasses/party-glasses.glb',
  g0:    '/models/glasses/glasses-classic.glb',
  g1:    '/models/glasses/glasses-1.glb',
  g2:    '/models/glasses/glasses-2.glb',
  g3:    '/models/glasses/glasses-3.glb',
  g4:    '/models/glasses/glasses-4.glb',
  g5:    '/models/glasses/glasses-5.glb',
  g6:    '/models/glasses/glasses-6.glb',
  g7:    '/models/glasses/glasses-7.glb',
  g8:    '/models/glasses/glasses-8.glb',
  g9:    '/models/glasses/glasses-9.glb',
  g10:   '/models/glasses/glasses-10.glb',
  g11:   '/models/glasses/glasses-11.glb',
  g12:   '/models/glasses/glasses-12.glb',
  g13:   '/models/glasses/glasses-13.glb',
  g14:   '/models/glasses/glasses-14.glb',
  g15:   '/models/glasses/glasses-15.glb',
  g16:   '/models/glasses/glasses-16.glb',
  g17:   '/models/glasses/glasses-17.glb',
  g18:   '/models/glasses/glasses-18.glb',
  g19:   '/models/glasses/glasses-19.glb',
  g20:   '/models/glasses/glasses-20.glb',
  g21:   '/models/glasses/glasses-21.glb',
  g22:   '/models/glasses/glasses-22.glb',
  g23:   '/models/glasses/glasses-23.glb',
  g24:   '/models/glasses/glasses-24.glb',
};

// MediaPipe face oval landmark indices (ordered, forms a closed polygon)
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

// Builds an orthonormal face coordinate frame from key landmarks.
// Returns the face's bridge position, IPD, quaternion, and basis vectors.
function faceFrame(landmarks: Landmark[], W: number, H: number) {
  // Left eye (from face's own perspective) → viewer's right in unmirrored feed
  const lEye = avg(landmarks, [33, 133], W, H);
  // Right eye (from face's own perspective) → viewer's left
  const rEye = avg(landmarks, [263, 362], W, H);
  const chin = toVec3(landmarks[152], W, H);
  const bridge = lEye.clone().add(rEye).multiplyScalar(0.5);
  const ipd = lEye.distanceTo(rEye);

  // Gram-Schmidt orthonormal basis
  const right = rEye.clone().sub(lEye).normalize();
  const tempUp = bridge.clone().sub(chin).normalize();
  const forward = new THREE.Vector3().crossVectors(right, tempUp).normalize();
  const up = new THREE.Vector3().crossVectors(forward, right).normalize();

  const rotMat = new THREE.Matrix4().makeBasis(right, up, forward);
  const quat = new THREE.Quaternion().setFromRotationMatrix(rotMat);

  return { bridge, ipd, quat, up, right, forward };
}

// ---------------------------------------------------------------------------
// Main class
// ---------------------------------------------------------------------------

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

  private earringMap: Record<EarringStyle, { L: THREE.Group; R: THREE.Group }>;
  private activeEarringL: THREE.Group;
  private activeEarringR: THREE.Group;
  private earringsEnabled = false;

  private necklaceMap: Record<NecklaceStyle, THREE.Group>;
  private activeNecklace: THREE.Group;
  private necklaceEnabled = false;

  private occluder: THREE.Mesh;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.sortObjects = true;
    this.renderer.autoClear = false;

    this.scene = new THREE.Scene();

    // Orthographic camera — frustum is updated every frame to match video size
    this.camera = new THREE.OrthographicCamera(-400, 400, 225, -225, -1000, 1000);
    this.camera.position.z = 500;

    // Lighting: warm key from upper-right, cool fill from lower-left
    const ambient = new THREE.AmbientLight(0xffffff, 0.65);
    const key = new THREE.DirectionalLight(0xfffbe8, 1.1);
    key.position.set(1, 2, 3);
    const fill = new THREE.DirectionalLight(0xd0e8ff, 0.35);
    fill.position.set(-1, -0.5, 1);
    this.scene.add(ambient, key, fill);

    // Face occluder — invisible mesh that writes depth so accessories can
    // correctly appear behind the face silhouette (e.g. temple arms behind ears).
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
    for (const g of Object.values(this.glassesMap)) {
      g.renderOrder = 1;
      g.visible = false;
    }
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

    for (const g of Object.values(this.glassesMap)) this.scene.add(g);
    this.loadGLBModels();
    this.loadHatModels();
    this.loadEarringModels();
    this.loadNecklaceModels();
  }

  private loadGLBModels(): void {
    const loader = new GLTFLoader();

    const load = (style: GlassesStyle, url: string): void => {
      loader.load(url, (gltf) => {
        const model = gltf.scene;

        // Compute bounding box before any transforms
        model.updateWorldMatrix(true, true);
        const box = new THREE.Box3().setFromObject(model);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());

        // Mark temple arms — meshes whose world x deviates >30% of half-width from center
        model.traverse(child => {
          if (child instanceof THREE.Mesh) {
            const wp = new THREE.Vector3();
            child.getWorldPosition(wp);
            const dx = wp.x - center.x;
            if (Math.abs(dx) > size.x * 0.30) {
              child.userData.side = dx > 0 ? 1 : -1;
            }
          }
        });

        // Center and scale so width = 3.0 IPD units
        model.position.sub(center);
        model.scale.setScalar(2.6 / Math.max(size.x, 0.001));

        this.glassesMap[style].add(model);
      }, undefined, () => { /* skip on error */ });
    };

    for (const [style, url] of Object.entries(GLB_URLS) as [GlassesStyle, string][]) {
      load(style, url);
    }
  }

  private loadHatModels(): void {
    const loader = new GLTFLoader();
    const load = (style: HatStyle, url: string): void => {
      loader.load(url, (gltf) => {
        const model = gltf.scene;
        model.updateWorldMatrix(true, true);
        const box = new THREE.Box3().setFromObject(model);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        // Pivot at bottom-center so hat sits on top of head
        model.position.set(-center.x, -(center.y - size.y / 2), -center.z);
        model.scale.setScalar(2.2 / Math.max(size.x, 0.001));
        this.hatMap[style].add(model);
      }, undefined, () => {});
    };
    for (const [style, url] of Object.entries(HAT_URLS) as [HatStyle, string][]) load(style, url);
  }

  private loadEarringModels(): void {
    const loader = new GLTFLoader();
    const load = (style: EarringStyle, url: string): void => {
      loader.load(url, (gltf) => {
        const model = gltf.scene;
        model.updateWorldMatrix(true, true);
        const box = new THREE.Box3().setFromObject(model);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        // Pivot at top so earring hangs down from ear lobe
        model.position.set(-center.x, -(center.y + size.y / 2), -center.z);
        model.scale.setScalar(0.55 / Math.max(size.y, 0.001));
        this.earringMap[style].L.add(model);
        const modelR = model.clone(true);
        modelR.scale.x *= -1;
        this.earringMap[style].R.add(modelR);
      }, undefined, () => {});
    };
    for (const [style, url] of Object.entries(EARRING_URLS) as [EarringStyle, string][]) load(style, url);
  }

  private loadNecklaceModels(): void {
    const loader = new GLTFLoader();
    const load = (style: NecklaceStyle, url: string): void => {
      loader.load(url, (gltf) => {
        const model = gltf.scene;
        model.updateWorldMatrix(true, true);
        const box = new THREE.Box3().setFromObject(model);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        // Rotate to face camera — poly.pizza necklaces are typically in the XZ plane
        model.rotation.x = -Math.PI / 2;
        model.updateWorldMatrix(true, true);
        const box2 = new THREE.Box3().setFromObject(model);
        const size2 = box2.getSize(new THREE.Vector3());
        const center2 = box2.getCenter(new THREE.Vector3());
        // Pivot at top-center (clasp), hangs down from chin
        model.position.set(-center2.x, -(center2.y + size2.y / 2), -center2.z);
        model.scale.multiplyScalar(3.0 / Math.max(size2.x, 0.001));
        this.necklaceMap[style].add(model);
      }, undefined, () => {});
    };
    for (const [style, url] of Object.entries(NECKLACE_URLS) as [NecklaceStyle, string][]) load(style, url);
  }

  private refreshOccluder(landmarks: Landmark[], W: number, H: number): void {
    const pts = FACE_OVAL.map(i => toVec3(landmarks[i], W, H));

    // Fan triangulation from centroid — creates a filled face silhouette polygon.
    const center = new THREE.Vector3();
    pts.forEach(p => center.add(p));
    center.divideScalar(pts.length);
    center.z -= 8; // Slightly behind the face surface so temples go behind it

    const verts: number[] = [];
    const n = pts.length;
    for (let i = 0; i < n; i++) {
      const a = pts[i];
      const b = pts[(i + 1) % n];
      verts.push(center.x, center.y, center.z, a.x, a.y, a.z, b.x, b.y, b.z);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    const old = this.occluder.geometry;
    this.occluder.geometry = geo;
    old.dispose();
  }

  // Call once per frame when landmarks are detected.
  // splitScreen: when true, renders only into the right half of a 2×W canvas
  update(landmarks: Landmark[], W: number, H: number, splitScreen = false): void {
    if (landmarks.length < 478) return;

    const canvasW = splitScreen ? W * 2 : W;
    this.renderer.setSize(canvasW, H, false);

    this.camera.left = -W / 2; this.camera.right = W / 2;
    this.camera.top = H / 2; this.camera.bottom = -H / 2;
    this.camera.updateProjectionMatrix();

    if (splitScreen) {
      // Restrict rendering to the right half of the canvas (the warped side).
      // WebGL origin is bottom-left, so y=0 maps to canvas bottom.
      this.renderer.setViewport(W, 0, W, H);
      this.renderer.setScissor(W, 0, W, H);
      this.renderer.setScissorTest(true);
    } else {
      this.renderer.setViewport(0, 0, W, H);
      this.renderer.setScissorTest(false);
    }

    const face = faceFrame(landmarks, W, H);
    this.refreshOccluder(landmarks, W, H);

    if (this.activeGlasses.visible) {
      this.activeGlasses.position.copy(face.bridge);
      this.activeGlasses.position.addScaledVector(face.forward, face.ipd * 0.14);
      this.activeGlasses.position.addScaledVector(face.up, -face.ipd * 0.12);
      this.activeGlasses.quaternion.copy(face.quat);
      this.activeGlasses.scale.setScalar(face.ipd);

      // Hide the temple that rotates behind the head based on face yaw.
      // face.right.z < 0  → face turned so that side=1 temple goes behind.
      // face.right.z > 0  → face turned so that side=-1 temple goes behind.
      const yaw = face.right.z;
      const YAW_THRESHOLD = 0.10;
      this.activeGlasses.traverse(child => {
        if ('side' in child.userData) {
          const s = child.userData.side as number;
          child.visible = !(s === 1 && yaw < -YAW_THRESHOLD) && !(s === -1 && yaw > YAW_THRESHOLD);
        }
      });
    }

    if (this.activeHat.visible) {
      const topHead = toVec3(landmarks[10], W, H);
      this.activeHat.position.copy(topHead);
      this.activeHat.position.addScaledVector(face.up, face.ipd * 0.20);   // push up above hairline
      this.activeHat.position.addScaledVector(face.forward, face.ipd * 0.04);
      this.activeHat.quaternion.copy(face.quat);
      this.activeHat.scale.setScalar(face.ipd);
    }

    // Landmark 234 = left ear side, 454 = right ear side (more lateral than 172/397)
    if (this.activeEarringL.visible) {
      this.activeEarringL.position.copy(toVec3(landmarks[234], W, H));
      this.activeEarringL.position.addScaledVector(face.up, -face.ipd * 0.05);
      this.activeEarringL.quaternion.copy(face.quat);
      this.activeEarringL.scale.setScalar(face.ipd);
      this.activeEarringR.position.copy(toVec3(landmarks[454], W, H));
      this.activeEarringR.position.addScaledVector(face.up, -face.ipd * 0.05);
      this.activeEarringR.quaternion.copy(face.quat);
      this.activeEarringR.scale.setScalar(face.ipd);
    }

    if (this.activeNecklace.visible) {
      const chin = toVec3(landmarks[152], W, H);
      this.activeNecklace.position.copy(chin);
      this.activeNecklace.position.addScaledVector(face.up, -face.ipd * 0.15);
      this.activeNecklace.position.addScaledVector(face.forward, face.ipd * 0.05);
      this.activeNecklace.quaternion.copy(face.quat);
      this.activeNecklace.scale.setScalar(face.ipd);
    }

    this.renderer.clear();
    this.renderer.render(this.scene, this.camera);
  }

  // Clear the WebGL canvas (call when no face is detected)
  clear(): void {
    this.renderer.setScissorTest(false);
    this.renderer.setViewport(
      0, 0,
      this.renderer.domElement.width,
      this.renderer.domElement.height,
    );
    this.renderer.clear();
  }

  setAccessories(glasses: boolean, hat: boolean, earrings: boolean, necklace = false): void {
    this.glassesEnabled = glasses;
    this.activeGlasses.visible = glasses;
    this.hatEnabled = hat;
    this.activeHat.visible = hat;
    this.earringsEnabled = earrings;
    this.activeEarringL.visible = earrings;
    this.activeEarringR.visible = earrings;
    this.necklaceEnabled = necklace;
    this.activeNecklace.visible = necklace;
    this.occluder.visible = glasses || hat || earrings || necklace;
  }

  setGlassesStyle(style: GlassesStyle): void {
    this.activeGlasses.visible = false;
    this.activeGlasses = this.glassesMap[style];
    this.activeGlasses.visible = this.glassesEnabled;
  }

  setHatStyle(style: HatStyle): void {
    this.activeHat.visible = false;
    this.activeHat = this.hatMap[style];
    this.activeHat.visible = this.hatEnabled;
  }

  setEarringStyle(style: EarringStyle): void {
    this.activeEarringL.visible = false;
    this.activeEarringR.visible = false;
    this.activeEarringL = this.earringMap[style].L;
    this.activeEarringR = this.earringMap[style].R;
    this.activeEarringL.visible = this.earringsEnabled;
    this.activeEarringR.visible = this.earringsEnabled;
  }

  setNecklaceStyle(style: NecklaceStyle): void {
    this.activeNecklace.visible = false;
    this.activeNecklace = this.necklaceMap[style];
    this.activeNecklace.visible = this.necklaceEnabled;
  }

  dispose(): void {
    this.scene.traverse(obj => {
      if (obj instanceof THREE.Mesh) {
        obj.geometry.dispose();
        if (Array.isArray(obj.material)) {
          obj.material.forEach(m => m.dispose());
        } else {
          obj.material.dispose();
        }
      }
    });
    this.renderer.dispose();
  }
}
