/**
 * Interactive fsaverage5 cortical surface viewer.
 *
 * The server exports the mesh as packed binary buffers (positions / normals /
 * indices), so the browser builds a BufferGeometry with zero parsing cost.
 * Per-vertex scalar maps arrive as raw float32 and are painted into a vertex
 * colour attribute — 20 484 values update in well under a frame.
 *
 * Correctness notes that matter scientifically:
 *   - Vertex ordering follows the server manifest's `hemi_order`, the same
 *     convention verified against TRIBE's implementation. The viewer never
 *     assumes L-then-R.
 *   - NaN vertices (medial wall, censored, unmapped parcels) render as inert
 *     grey rather than as the bottom of the colour ramp, so "no data" can never
 *     be misread as "strong negative effect".
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { api } from '../../lib/api';

// ------------------------------------------------------------------ types

export type ColorMap = 'diverging' | 'agreement' | 'deviation';

interface Manifest {
  space: string;
  hemi_order: string[];
  source: string;
  total_vertices: number;
  hemispheres: Record<
    string,
    { n_vertices: number; n_faces: number; vertex_offset: number }
  >;
}

interface Props {
  /** Endpoint returning float32 per-vertex values (one per surface vertex). */
  valuesUrl: string | null;
  colorMap?: ColorMap;
  /** Fixed colour range; omit to scale robustly from the data. */
  domain?: [number, number];
  legendLabel?: string;
  height?: number;
  showControls?: boolean;
  onVertexPick?: (vertex: number, value: number) => void;
}

// ------------------------------------------------------------------ colour

/** Perceptually ordered ramps sampled to a 256-entry lookup texture. */
const RAMPS: Record<ColorMap, [number, number, number][]> = {
  // Blue → neutral → magenta. Symmetric: sign is meaningful.
  diverging: [
    [0.13, 0.35, 0.78], [0.35, 0.58, 0.9], [0.68, 0.78, 0.92],
    [0.88, 0.88, 0.88],
    [0.95, 0.74, 0.85], [0.88, 0.42, 0.72], [0.72, 0.15, 0.52],
  ],
  // Dark → cyan → white. Sequential: higher agreement is brighter.
  agreement: [
    [0.06, 0.08, 0.17], [0.09, 0.24, 0.45], [0.11, 0.45, 0.65],
    [0.25, 0.68, 0.75], [0.6, 0.86, 0.85], [0.94, 0.98, 0.98],
  ],
  // Dark → amber → white. Sequential: more deviation is hotter.
  deviation: [
    [0.08, 0.06, 0.14], [0.32, 0.11, 0.32], [0.62, 0.2, 0.28],
    [0.85, 0.44, 0.16], [0.96, 0.72, 0.28], [0.99, 0.95, 0.78],
  ],
};

/** Colour used where a vertex has no value. Deliberately outside every ramp. */
const NO_DATA: [number, number, number] = [0.28, 0.29, 0.33];

function sampleRamp(ramp: [number, number, number][], t: number): [number, number, number] {
  const clamped = Math.min(1, Math.max(0, t));
  const scaled = clamped * (ramp.length - 1);
  const index = Math.min(ramp.length - 2, Math.floor(scaled));
  const frac = scaled - index;
  const a = ramp[index];
  const b = ramp[index + 1];
  return [
    a[0] + (b[0] - a[0]) * frac,
    a[1] + (b[1] - a[1]) * frac,
    a[2] + (b[2] - a[2]) * frac,
  ];
}

/** Robust 2nd–98th percentile domain, symmetric for diverging maps. */
function robustDomain(values: Float32Array, diverging: boolean): [number, number] {
  const finite: number[] = [];
  for (let i = 0; i < values.length; i += 1) {
    if (Number.isFinite(values[i])) finite.push(values[i]);
  }
  if (finite.length === 0) return [0, 1];
  finite.sort((a, b) => a - b);
  const low = finite[Math.floor(finite.length * 0.02)];
  const high = finite[Math.floor(finite.length * 0.98)];
  if (diverging) {
    const extent = Math.max(Math.abs(low), Math.abs(high)) || 1;
    return [-extent, extent];
  }
  return low === high ? [low, low + 1] : [low, high];
}

// ------------------------------------------------------------------ view

const VIEWS = {
  lateral: { azimuth: -Math.PI / 2, polar: Math.PI / 2 },
  medial: { azimuth: Math.PI / 2, polar: Math.PI / 2 },
  dorsal: { azimuth: 0, polar: 0.12 },
  ventral: { azimuth: 0, polar: Math.PI - 0.12 },
  anterior: { azimuth: 0, polar: Math.PI / 2 },
  posterior: { azimuth: Math.PI, polar: Math.PI / 2 },
} as const;

type ViewName = keyof typeof VIEWS;
type HemiMode = 'both' | 'L' | 'R';

export default function CorticalViewer({
  valuesUrl,
  colorMap = 'diverging',
  domain,
  legendLabel = 'value',
  height = 480,
  showControls = true,
  onVertexPick,
}: Props) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    meshes: Record<string, THREE.Mesh>;
    group: THREE.Group;
    dispose: () => void;
  } | null>(null);

  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [values, setValues] = useState<Float32Array | null>(null);
  const [status, setStatus] = useState<'loading' | 'ready' | 'error' | 'empty'>('loading');
  const [message, setMessage] = useState<string>('Loading cortical surface…');
  const [view, setView] = useState<ViewName>('lateral');
  const [hemi, setHemi] = useState<HemiMode>('both');
  const [hover, setHover] = useState<{ vertex: number; value: number; x: number; y: number } | null>(null);

  const activeDomain = useMemo<[number, number]>(() => {
    if (domain) return domain;
    if (!values) return [0, 1];
    return robustDomain(values, colorMap === 'diverging');
  }, [domain, values, colorMap]);

  // ---------------------------------------------------------------- scene
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let disposed = false;
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    mount.appendChild(renderer.domElement);
    renderer.domElement.style.touchAction = 'none';
    renderer.domElement.setAttribute('role', 'img');
    renderer.domElement.setAttribute('aria-label', `Cortical surface, ${legendLabel}`);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      38, mount.clientWidth / mount.clientHeight, 1, 3000,
    );
    camera.position.set(0, 0, 420);

    const group = new THREE.Group();
    scene.add(group);

    // Three-point lighting: readable shading without washing out the colour map.
    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(1, 1.2, 1.4);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x9fbaff, 0.65);
    fill.position.set(-1.2, -0.4, 0.8);
    scene.add(fill);
    const rim = new THREE.DirectionalLight(0xffb0e8, 0.5);
    rim.position.set(0, 0.5, -1.5);
    scene.add(rim);

    // -- orbit state (hand-rolled: no extra dependency, exact behaviour we want)
    const spherical = new THREE.Spherical(420, Math.PI / 2, -Math.PI / 2);
    const target = new THREE.Vector3(0, 0, 0);
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    let autoRotate = true;

    const applyCamera = () => {
      spherical.phi = Math.max(0.05, Math.min(Math.PI - 0.05, spherical.phi));
      spherical.radius = Math.max(150, Math.min(1200, spherical.radius));
      camera.position.setFromSpherical(spherical).add(target);
      camera.lookAt(target);
    };
    applyCamera();

    const onPointerDown = (event: PointerEvent) => {
      dragging = true;
      autoRotate = false;
      lastX = event.clientX;
      lastY = event.clientY;
      (event.target as Element).setPointerCapture?.(event.pointerId);
    };
    const onPointerUp = (event: PointerEvent) => {
      dragging = false;
      (event.target as Element).releasePointerCapture?.(event.pointerId);
    };
    const onPointerMove = (event: PointerEvent) => {
      if (dragging) {
        spherical.theta -= (event.clientX - lastX) * 0.0075;
        spherical.phi -= (event.clientY - lastY) * 0.0075;
        lastX = event.clientX;
        lastY = event.clientY;
        applyCamera();
      }
    };
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      autoRotate = false;
      spherical.radius *= 1 + Math.sign(event.deltaY) * 0.09;
      applyCamera();
    };

    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointerup', onPointerUp);
    window.addEventListener('pointermove', onPointerMove);
    renderer.domElement.addEventListener('wheel', onWheel, { passive: false });

    // -- picking
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const onPick = (event: PointerEvent) => {
      if (dragging) return;
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(group.children, false);
      const hit = hits[0];
      if (!hit || hit.face === undefined || hit.face === null) {
        setHover(null);
        return;
      }
      const offset = (hit.object.userData.vertexOffset as number) ?? 0;
      const vertex = offset + hit.face.a;
      const array = (hit.object.userData.values as Float32Array | undefined);
      const value = array ? array[vertex] : Number.NaN;
      setHover({ vertex, value, x: event.clientX - rect.left, y: event.clientY - rect.top });
    };
    renderer.domElement.addEventListener('pointermove', onPick);
    renderer.domElement.addEventListener('pointerleave', () => setHover(null));
    renderer.domElement.addEventListener('click', (event) => {
      onPick(event as unknown as PointerEvent);
      if (hover && onVertexPick) onVertexPick(hover.vertex, hover.value);
    });

    const resize = () => {
      if (!mount.clientWidth) return;
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(mount);

    let frame = 0;
    const animate = () => {
      if (disposed) return;
      frame = requestAnimationFrame(animate);
      if (autoRotate) {
        spherical.theta += 0.0016;
        applyCamera();
      }
      renderer.render(scene, camera);
    };
    animate();

    const dispose = () => {
      disposed = true;
      cancelAnimationFrame(frame);
      observer.disconnect();
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      renderer.domElement.removeEventListener('wheel', onWheel);
      renderer.domElement.removeEventListener('pointermove', onPick);
      window.removeEventListener('pointerup', onPointerUp);
      window.removeEventListener('pointermove', onPointerMove);
      group.children.forEach((child) => {
        const mesh = child as THREE.Mesh;
        mesh.geometry.dispose();
        (mesh.material as THREE.Material).dispose();
      });
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
    };

    sceneRef.current = { renderer, scene, camera, meshes: {}, group, dispose };
    return dispose;
    // Deliberately mount-once: geometry and values are updated by later effects.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---------------------------------------------------------------- geometry
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const loaded = await api.get<Manifest>('/surface/manifest');
        if (cancelled) return;
        setManifest(loaded);

        const context = sceneRef.current;
        if (!context) return;

        for (const hemisphere of loaded.hemi_order) {
          const [positions, normals, indices] = await Promise.all([
            api.buffer(`/surface/${hemisphere}/positions`),
            api.buffer(`/surface/${hemisphere}/normals`),
            api.buffer(`/surface/${hemisphere}/indices`),
          ]);
          if (cancelled) return;

          const geometry = new THREE.BufferGeometry();
          geometry.setAttribute(
            'position', new THREE.BufferAttribute(new Float32Array(positions.data), 3),
          );
          geometry.setAttribute(
            'normal', new THREE.BufferAttribute(new Float32Array(normals.data), 3),
          );
          geometry.setIndex(
            new THREE.BufferAttribute(new Uint32Array(indices.data), 1),
          );

          const nVertices = loaded.hemispheres[hemisphere].n_vertices;
          const colors = new Float32Array(nVertices * 3);
          colors.fill(0.55);
          geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
          geometry.computeBoundingSphere();

          const material = new THREE.MeshStandardMaterial({
            vertexColors: true,
            roughness: 0.62,
            metalness: 0.06,
            flatShading: false,
            side: THREE.DoubleSide,
          });

          const mesh = new THREE.Mesh(geometry, material);
          // Separate the hemispheres slightly so the medial walls stay visible.
          mesh.position.x = hemisphere === 'L' ? -6 : 6;
          mesh.userData.vertexOffset = loaded.hemispheres[hemisphere].vertex_offset;
          mesh.userData.hemi = hemisphere;
          context.group.add(mesh);
          context.meshes[hemisphere] = mesh;
        }

        if (!cancelled) {
          setStatus(valuesUrl ? 'loading' : 'empty');
          setMessage(valuesUrl ? 'Loading map…' : 'No map selected');
        }
      } catch (error) {
        if (!cancelled) {
          setStatus('error');
          setMessage(
            `Could not load the cortical surface: ${(error as Error).message}`,
          );
        }
      }
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---------------------------------------------------------------- values
  useEffect(() => {
    if (!valuesUrl || !manifest) {
      if (!valuesUrl) {
        setStatus('empty');
        setMessage('No map selected');
      }
      return;
    }
    let cancelled = false;
    setStatus('loading');
    setMessage('Loading map…');

    (async () => {
      try {
        const { data } = await api.buffer(valuesUrl);
        if (cancelled) return;
        const array = new Float32Array(data);
        if (array.length !== manifest.total_vertices) {
          setStatus('error');
          setMessage(
            `Map has ${array.length} values but the ${manifest.space} surface has ` +
              `${manifest.total_vertices} vertices. Refusing to render a mismatched map.`,
          );
          return;
        }
        setValues(array);
        setStatus('ready');
      } catch (error) {
        if (!cancelled) {
          setStatus('error');
          setMessage(`Map unavailable: ${(error as Error).message}`);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [valuesUrl, manifest]);

  // ---------------------------------------------------------------- paint
  useEffect(() => {
    const context = sceneRef.current;
    if (!context || !manifest || !values) return;

    const ramp = RAMPS[colorMap];
    const [low, high] = activeDomain;
    const span = high - low || 1;

    for (const hemisphere of manifest.hemi_order) {
      const mesh = context.meshes[hemisphere];
      if (!mesh) continue;
      const info = manifest.hemispheres[hemisphere];
      const attribute = mesh.geometry.getAttribute('color') as THREE.BufferAttribute;
      const colors = attribute.array as Float32Array;

      for (let i = 0; i < info.n_vertices; i += 1) {
        const value = values[info.vertex_offset + i];
        let rgb: [number, number, number];
        if (!Number.isFinite(value)) {
          rgb = NO_DATA;
        } else {
          rgb = sampleRamp(ramp, (value - low) / span);
        }
        colors[i * 3] = rgb[0];
        colors[i * 3 + 1] = rgb[1];
        colors[i * 3 + 2] = rgb[2];
      }
      attribute.needsUpdate = true;
      mesh.userData.values = values;
    }
  }, [values, manifest, colorMap, activeDomain]);

  // ---------------------------------------------------------------- view/hemi
  useEffect(() => {
    const context = sceneRef.current;
    if (!context) return;
    const preset = VIEWS[view];
    const spherical = new THREE.Spherical(
      context.camera.position.length(), preset.polar, preset.azimuth,
    );
    context.camera.position.setFromSpherical(spherical);
    context.camera.lookAt(0, 0, 0);
  }, [view]);

  useEffect(() => {
    const context = sceneRef.current;
    if (!context || !manifest) return;
    for (const hemisphere of manifest.hemi_order) {
      const mesh = context.meshes[hemisphere];
      if (mesh) mesh.visible = hemi === 'both' || hemi === hemisphere;
    }
  }, [hemi, manifest]);

  // ---------------------------------------------------------------- render
  const legendStops = useMemo(() => {
    const ramp = RAMPS[colorMap];
    return Array.from({ length: 24 }, (_, index) => {
      const [r, g, b] = sampleRamp(ramp, index / 23);
      return `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
    }).join(',');
  }, [colorMap]);

  return (
    <div className="panel relative overflow-hidden">
      <div ref={mountRef} style={{ height, width: '100%' }} className="relative" />

      {status !== 'ready' && (
        <div className="absolute inset-0 grid place-items-center px-6 text-center">
          <div className="max-w-md">
            {status === 'loading' && (
              <div
                className="mx-auto mb-3 h-7 w-7 rounded-full border-2 border-transparent animate-spin-slow"
                style={{ borderTopColor: 'var(--color-signal)', borderRightColor: 'var(--color-signal)' }}
              />
            )}
            <p
              className="text-sm"
              style={{ color: status === 'error' ? 'var(--color-alarm)' : 'var(--fg-muted)' }}
            >
              {message}
            </p>
          </div>
        </div>
      )}

      {hover && Number.isFinite(hover.value) && (
        <div
          className="pointer-events-none absolute z-10 rounded-lg px-2.5 py-1.5 text-[0.72rem] numeric glass"
          style={{ left: Math.min(hover.x + 14, 600), top: hover.y + 14 }}
        >
          <div style={{ color: 'var(--fg-muted)' }}>vertex {hover.vertex}</div>
          <div className="font-semibold">{hover.value.toFixed(4)}</div>
        </div>
      )}

      {showControls && (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 flex flex-wrap items-end justify-between gap-3 p-3">
          <div className="pointer-events-auto flex flex-wrap gap-1.5">
            <div className="glass flex gap-0.5 rounded-lg p-0.5">
              {(['both', 'L', 'R'] as HemiMode[]).map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setHemi(option)}
                  aria-pressed={hemi === option}
                  className="rounded-md px-2 py-1 text-[0.7rem] font-semibold transition-colors"
                  style={
                    hemi === option
                      ? { background: 'var(--color-signal)', color: 'white' }
                      : { color: 'var(--fg-muted)' }
                  }
                >
                  {option === 'both' ? 'L+R' : option}
                </button>
              ))}
            </div>
            <div className="glass flex flex-wrap gap-0.5 rounded-lg p-0.5">
              {(Object.keys(VIEWS) as ViewName[]).map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setView(option)}
                  aria-pressed={view === option}
                  className="rounded-md px-2 py-1 text-[0.7rem] font-medium capitalize transition-colors"
                  style={
                    view === option
                      ? { background: 'var(--color-signal)', color: 'white' }
                      : { color: 'var(--fg-muted)' }
                  }
                >
                  {option}
                </button>
              ))}
            </div>
          </div>

          {status === 'ready' && (
            <div className="pointer-events-auto glass rounded-lg px-2.5 py-2">
              <div className="mb-1 text-[0.62rem] font-semibold uppercase tracking-[0.09em]" style={{ color: 'var(--fg-muted)' }}>
                {legendLabel}
              </div>
              <div
                className="h-2 w-40 rounded-full"
                style={{ background: `linear-gradient(90deg, ${legendStops})` }}
              />
              <div className="mt-1 flex justify-between text-[0.64rem] numeric" style={{ color: 'var(--fg-muted)' }}>
                <span>{activeDomain[0].toFixed(2)}</span>
                <span>{activeDomain[1].toFixed(2)}</span>
              </div>
              <div className="mt-1.5 flex items-center gap-1.5 text-[0.62rem]" style={{ color: 'var(--fg-faint)' }}>
                <span
                  className="inline-block h-2 w-2 rounded-sm"
                  style={{ background: `rgb(${NO_DATA.map((c) => Math.round(c * 255)).join(',')})` }}
                />
                no data / medial wall
              </div>
            </div>
          )}
        </div>
      )}

      {manifest && (
        <div className="absolute right-3 top-3 text-[0.62rem] numeric" style={{ color: 'var(--fg-faint)' }}>
          {manifest.space} · {manifest.total_vertices.toLocaleString()} vertices ·{' '}
          {manifest.hemi_order.join('|')}
        </div>
      )}
    </div>
  );
}
