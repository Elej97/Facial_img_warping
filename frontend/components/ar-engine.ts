import * as THREE from 'three';

type Landmark = { x: number; y: number; z: number };

export type GlassesStyle = 'classic' | 'round' | 'aviator' | 'square';

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
// Geometry builders — all sized in "IPD units" (1 unit = inter-pupillary distance).
// The AREngine scales each group by the actual pixel IPD before rendering.
// ---------------------------------------------------------------------------

function buildGlasses(): THREE.Group {
  const g = new THREE.Group();

  const frameMat = new THREE.MeshStandardMaterial({ color: 0x111122, metalness: 0.9, roughness: 0.15 });
  const lensMat = new THREE.MeshStandardMaterial({
    color: 0x4499bb, metalness: 0.05, roughness: 0.0,
    transparent: true, opacity: 0.28,
  });

  const HR = 0.30;   // horizontal torus radius
  const VR = 0.24;   // vertical scale (gives oval lens)
  const TR = 0.017;  // tube radius (frame wire thickness)
  const CX = 0.48;   // lens center x offset from bridge

  for (const side of [-1, 1] as const) {
    const frame = new THREE.Mesh(new THREE.TorusGeometry(HR, TR, 8, 48), frameMat);
    frame.scale.y = VR / HR;
    frame.position.set(side * CX, 0, 0);
    g.add(frame);

    const lens = new THREE.Mesh(new THREE.CircleGeometry(HR - TR * 0.5, 48), lensMat);
    lens.scale.y = VR / HR;
    lens.position.set(side * CX, 0, 0.002);
    g.add(lens);
  }

  // Nose bridge
  const bridgeLen = 2 * CX - 2 * HR + 0.08;
  const bridgeMesh = new THREE.Mesh(
    new THREE.CylinderGeometry(TR * 0.75, TR * 0.75, bridgeLen, 8),
    frameMat,
  );
  bridgeMesh.rotation.z = Math.PI / 2;
  bridgeMesh.position.set(0, 0.03, 0.01);
  g.add(bridgeMesh);

  // Temple arms — angled outward and backward toward ears
  const tLen = 1.15;
  for (const side of [-1, 1] as const) {
    const temple = new THREE.Mesh(
      new THREE.CylinderGeometry(TR * 0.65, TR * 0.65, tLen, 8),
      frameMat,
    );
    const dir = new THREE.Vector3(side, -0.05, -0.28).normalize();
    temple.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    temple.position.set(side * (CX + HR - 0.02), 0.015, 0).addScaledVector(dir, tLen / 2);
    temple.userData.side = side;
    g.add(temple);
  }

  return g;
}

function buildRoundGlasses(): THREE.Group {
  const g = new THREE.Group();
  const frameMat = new THREE.MeshStandardMaterial({ color: 0xd4a030, metalness: 0.95, roughness: 0.1 });
  const lensMat = new THREE.MeshStandardMaterial({ color: 0x88ccbb, transparent: true, opacity: 0.22, roughness: 0 });
  const R = 0.27; const TR = 0.013; const CX = 0.46;
  for (const side of [-1, 1] as const) {
    const frame = new THREE.Mesh(new THREE.TorusGeometry(R, TR, 8, 48), frameMat);
    frame.position.set(side * CX, 0, 0);
    g.add(frame);
    const lens = new THREE.Mesh(new THREE.CircleGeometry(R - TR, 48), lensMat);
    lens.position.set(side * CX, 0, 0.002);
    g.add(lens);
  }
  const bridgeLen = 2 * CX - 2 * R + 0.06;
  const bridge = new THREE.Mesh(new THREE.CylinderGeometry(TR * 0.7, TR * 0.7, bridgeLen, 8), frameMat);
  bridge.rotation.z = Math.PI / 2;
  bridge.position.set(0, 0.05, 0.01);
  g.add(bridge);
  for (const side of [-1, 1] as const) {
    const temple = new THREE.Mesh(new THREE.CylinderGeometry(TR * 0.6, TR * 0.6, 1.1, 8), frameMat);
    const dir = new THREE.Vector3(side, -0.04, -0.28).normalize();
    temple.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    temple.position.set(side * (CX + R - 0.01), 0.01, 0).addScaledVector(dir, 1.1 / 2);
    temple.userData.side = side;
    g.add(temple);
  }
  return g;
}

function buildAviatorGlasses(): THREE.Group {
  const g = new THREE.Group();
  const frameMat = new THREE.MeshStandardMaterial({ color: 0xb0b8c8, metalness: 0.95, roughness: 0.08 });
  const lensMat = new THREE.MeshStandardMaterial({ color: 0x667788, transparent: true, opacity: 0.40, roughness: 0 });
  const HR = 0.27; const VR = 0.34; const TR = 0.013; const CX = 0.50;
  for (const side of [-1, 1] as const) {
    const frame = new THREE.Mesh(new THREE.TorusGeometry(HR, TR, 8, 48), frameMat);
    frame.scale.y = VR / HR;
    frame.position.set(side * CX, -0.04, 0);
    g.add(frame);
    const lens = new THREE.Mesh(new THREE.CircleGeometry(HR - TR, 48), lensMat);
    lens.scale.y = VR / HR;
    lens.position.set(side * CX, -0.04, 0.002);
    g.add(lens);
  }
  const bridgeLen = 2 * CX - 2 * HR + 0.04;
  const bridge = new THREE.Mesh(new THREE.CylinderGeometry(TR * 0.6, TR * 0.6, bridgeLen, 8), frameMat);
  bridge.rotation.z = Math.PI / 2;
  bridge.position.set(0, 0.05, 0.01);
  g.add(bridge);
  for (const side of [-1, 1] as const) {
    const temple = new THREE.Mesh(new THREE.CylinderGeometry(TR * 0.55, TR * 0.55, 1.1, 8), frameMat);
    const dir = new THREE.Vector3(side, -0.04, -0.28).normalize();
    temple.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    temple.position.set(side * (CX + HR - 0.01), -0.02, 0).addScaledVector(dir, 1.1 / 2);
    temple.userData.side = side;
    g.add(temple);
  }
  return g;
}

function buildSquareGlasses(): THREE.Group {
  const g = new THREE.Group();
  const frameMat = new THREE.MeshStandardMaterial({ color: 0x1a2233, metalness: 0.6, roughness: 0.35 });
  const lensMat = new THREE.MeshStandardMaterial({ color: 0x334455, transparent: true, opacity: 0.28, roughness: 0 });
  const W = 0.30; const H = 0.21; const TR = 0.016; const CX = 0.48;
  for (const side of [-1, 1] as const) {
    const lens = new THREE.Mesh(new THREE.PlaneGeometry(W * 2, H * 2), lensMat);
    lens.position.set(side * CX, 0, 0.001);
    g.add(lens);
    for (const [cy, isH] of [[H, true], [-H, true]] as [number, boolean][]) {
      const bar = new THREE.Mesh(new THREE.CylinderGeometry(TR, TR, W * 2 + TR * 2, 8), frameMat);
      bar.rotation.z = Math.PI / 2;
      bar.position.set(side * CX, cy, 0.002);
      g.add(bar);
    }
    for (const cx of [-W, W]) {
      const bar = new THREE.Mesh(new THREE.CylinderGeometry(TR, TR, H * 2, 8), frameMat);
      bar.position.set(side * CX + cx, 0, 0.002);
      g.add(bar);
    }
  }
  const bridgeLen = Math.max(0.01, 2 * CX - 2 * W - TR * 2);
  const bridge = new THREE.Mesh(new THREE.CylinderGeometry(TR * 0.75, TR * 0.75, bridgeLen, 8), frameMat);
  bridge.rotation.z = Math.PI / 2;
  bridge.position.set(0, H * 0.3, 0.002);
  g.add(bridge);
  for (const side of [-1, 1] as const) {
    const temple = new THREE.Mesh(new THREE.CylinderGeometry(TR * 0.65, TR * 0.65, 1.1, 8), frameMat);
    const dir = new THREE.Vector3(side, -0.04, -0.28).normalize();
    temple.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    temple.position.set(side * (CX + W), H * 0.1, 0).addScaledVector(dir, 1.1 / 2);
    temple.userData.side = side;
    g.add(temple);
  }
  return g;
}

function buildHat(): THREE.Group {
  const g = new THREE.Group();

  const crownMat = new THREE.MeshStandardMaterial({ color: 0x1a0f00, roughness: 0.88 });
  const brimMat = new THREE.MeshStandardMaterial({ color: 0x261500, roughness: 0.82, side: THREE.DoubleSide });
  const bandMat = new THREE.MeshStandardMaterial({ color: 0x8b0000, roughness: 0.55 });

  // In IPD units:
  const CR = 0.38;   // crown base radius
  const CH = 0.62;   // crown height
  const BR = 0.82;   // brim outer radius

  const crown = new THREE.Mesh(new THREE.CylinderGeometry(CR * 0.90, CR, CH, 32), crownMat);
  crown.position.y = CH / 2;
  g.add(crown);

  // Top cap (CylinderGeometry already includes caps, but add explicit one for slight overhang look)
  const cap = new THREE.Mesh(new THREE.CircleGeometry(CR * 0.90, 32), crownMat);
  cap.rotation.x = -Math.PI / 2;
  cap.position.y = CH + 0.001;
  g.add(cap);

  // Brim (flat ring)
  const brim = new THREE.Mesh(new THREE.RingGeometry(CR, BR, 48), brimMat);
  brim.rotation.x = -Math.PI / 2;
  brim.position.y = 0.005;
  g.add(brim);

  // Hat band at crown base
  const band = new THREE.Mesh(new THREE.CylinderGeometry(CR + 0.007, CR + 0.007, 0.075, 32), bandMat);
  band.position.y = 0.05;
  g.add(band);

  return g;
}

function buildEarring(): THREE.Group {
  const g = new THREE.Group();

  const goldMat = new THREE.MeshStandardMaterial({ color: 0xffd700, metalness: 1.0, roughness: 0.08 });
  const gemMat = new THREE.MeshStandardMaterial({ color: 0xff69b4, metalness: 0.2, roughness: 0.05 });

  // Sized in "earring units" — scaled externally to ~0.28 × IPD
  const hoop = new THREE.Mesh(new THREE.TorusGeometry(0.3, 0.065, 8, 32), goldMat);
  g.add(hoop);

  const wire = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.17, 6), goldMat);
  wire.position.y = -0.22;
  g.add(wire);

  const gem = new THREE.Mesh(new THREE.SphereGeometry(0.1, 10, 10), gemMat);
  gem.position.y = -0.42;
  g.add(gem);

  return g;
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
  private hat: THREE.Group;
  private leftEarring: THREE.Group;
  private rightEarring: THREE.Group;
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

    this.glassesMap = {
      classic: buildGlasses(),
      round: buildRoundGlasses(),
      aviator: buildAviatorGlasses(),
      square: buildSquareGlasses(),
    };
    for (const g of Object.values(this.glassesMap)) {
      g.renderOrder = 1;
      g.visible = false;
    }
    this.activeGlasses = this.glassesMap.classic;

    this.hat = buildHat();
    this.hat.renderOrder = 1;
    this.hat.visible = false;

    this.leftEarring = buildEarring();
    this.leftEarring.renderOrder = 1;
    this.leftEarring.visible = false;

    this.rightEarring = buildEarring();
    this.rightEarring.renderOrder = 1;
    this.rightEarring.visible = false;

    for (const g of Object.values(this.glassesMap)) this.scene.add(g);
    this.scene.add(this.hat, this.leftEarring, this.rightEarring);
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
      this.activeGlasses.quaternion.copy(face.quat);
      this.activeGlasses.scale.setScalar(face.ipd);

      // Hide the temple that rotates behind the head based on face yaw.
      // face.right.z < 0  → face turned so that side=1 temple goes behind.
      // face.right.z > 0  → face turned so that side=-1 temple goes behind.
      const yaw = face.right.z;
      const YAW_THRESHOLD = 0.10;
      this.activeGlasses.children.forEach(child => {
        if ('side' in child.userData) {
          const s = child.userData.side as number;
          child.visible = !(s === 1 && yaw < -YAW_THRESHOLD) && !(s === -1 && yaw > YAW_THRESHOLD);
        }
      });
    }

    if (this.hat.visible) {
      const topHead = toVec3(landmarks[10], W, H);
      this.hat.position.copy(topHead);
      this.hat.position.addScaledVector(face.forward, face.ipd * 0.04);
      this.hat.quaternion.copy(face.quat);
      this.hat.scale.setScalar(face.ipd * 0.88);
    }

    // Landmark 172 = left ear lobe (face-left = viewer-right in unmirrored feed)
    if (this.leftEarring.visible) {
      this.leftEarring.position.copy(toVec3(landmarks[172], W, H));
      this.leftEarring.quaternion.copy(face.quat);
      this.leftEarring.scale.setScalar(face.ipd * 0.28);
    }

    // Landmark 397 = right ear lobe
    if (this.rightEarring.visible) {
      this.rightEarring.position.copy(toVec3(landmarks[397], W, H));
      this.rightEarring.quaternion.copy(face.quat);
      this.rightEarring.scale.setScalar(face.ipd * 0.28);
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

  setAccessories(glasses: boolean, hat: boolean, earrings: boolean): void {
    this.glassesEnabled = glasses;
    this.activeGlasses.visible = glasses;
    this.hat.visible = hat;
    this.leftEarring.visible = earrings;
    this.rightEarring.visible = earrings;
    this.occluder.visible = glasses || hat || earrings;
  }

  setGlassesStyle(style: GlassesStyle): void {
    this.activeGlasses.visible = false;
    this.activeGlasses = this.glassesMap[style];
    this.activeGlasses.visible = this.glassesEnabled;
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
