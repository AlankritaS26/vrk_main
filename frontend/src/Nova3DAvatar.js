import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { clone as skeletonClone } from 'three/examples/jsm/utils/SkeletonUtils.js';

/**
 * Nova3DAvatar
 * Half-body Ready Player Me character with a processing "thinking" pose
 * (right hand to temple) and studio lighting.
 */

const VISEME_SEQUENCE = [
  'viseme_aa', 'viseme_E', 'viseme_I', 'viseme_O', 'viseme_U',
  'viseme_PP', 'viseme_FF', 'viseme_TH', 'viseme_DD', 'viseme_kk',
  'viseme_SS', 'jawOpen',
];

let _cachedGLTF  = null;
let _loadPromise = null;

function preloadNova() {
  if (_cachedGLTF || _loadPromise) return _loadPromise || Promise.resolve(_cachedGLTF);
  _loadPromise = new Promise((resolve, reject) => {
    new GLTFLoader().load('/nova_avatar.glb', gltf => { _cachedGLTF = gltf; resolve(gltf); }, undefined, reject);
  });
  return _loadPromise;
}

preloadNova();

function styleMaterial(mesh) {
  const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
  mats.forEach(mat => {
    if (!mat) return;
    const name = `${mesh.name || ''} ${mat.name || ''}`;
    if (mat.map) mat.map.colorSpace = THREE.SRGBColorSpace;
    if (mat.normalMap) mat.normalMap.colorSpace = THREE.NoColorSpace;
    mat.envMapIntensity = 0.85;
    if (/Head|Body|Teeth/i.test(name)) {
      mat.roughness = Math.min(Math.max(mat.roughness ?? 0.5, 0.42), 0.58);
      mat.metalness = 0.02;
    } else if (/Hair/i.test(name)) {
      mat.roughness = 0.52;
      mat.metalness = 0.06;
    } else if (/Eye/i.test(name)) {
      mat.roughness = 0.12;
      mat.metalness = 0.05;
    } else if (/Glasses/i.test(name)) {
      mat.roughness = 0.08;
      mat.metalness = 0.35;
      mat.transparent = true;
      mat.opacity = Math.min(mat.opacity ?? 1, 0.88);
    } else if (/Wolf3D_Outfit_Top|Blazer|Jacket|Dress|Uniform/i.test(name)) {
      // The asset has one textured top surface rather than a separate blazer mesh.
      // Remove the texture tinting so the blue outerwear is visibly applied.
      mat.map = null;
      mat.color.set(0x111318);
      mat.roughness = 0.62;
      mat.metalness = 0.02;
    } else if (/Collar|Cuff|Shirt|Blouse|Lapels/i.test(name)) {
      mat.color.set(0xf7f4ec);
      mat.roughness = 0.72;
    } else if (/Outfit_Bottom|Bottom|Skirt|Pants|Footwear/i.test(name)) {
      mat.color.set(0x111827);
      mat.roughness = 0.62;
      mat.metalness = 0.02;
    } else if (/Outfit|Top|Bottom|Footwear/i.test(name)) {
      mat.roughness = Math.min(mat.roughness ?? 0.7, 0.78);
    }
    mat.needsUpdate = true;
  });
}

export default function Nova3DAvatar({ st = 'idle', size = { width: '100%', height: '560px' } }) {
  const containerRef = useRef(null);
  const [loading, setLoading]     = useState(true);
  const [loadError, setLoadError] = useState(null);

  const statusRef = useRef(st);
  useEffect(() => { statusRef.current = st; }, [st]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let animationFrameId;
    let renderer, scene, camera;
    let isDisposed = false;
    let modelRoot = null;

    const morphMeshes = [];
    const eyeObjects = [];
    let headBone = null, neckBone = null, spineBone = null, spine1Bone = null;
    let rightArmBone = null, rightForeArmBone = null, rightHandBone = null;
    let leftArmBone = null, leftForeArmBone = null, leftHandBone = null;
    let restRightArm = null, restRightForeArm = null, restRightHand = null;
    let restLeftArm = null, restLeftForeArm = null, restLeftHand = null;
    const fingerBones = [];
    let restHeadRotX = 0, restHeadRotY = 0, restHeadRotZ = 0;
    let restNeckRotX = 0, restNeckRotY = 0;
    let restSpineRotX = 0, restSpine1RotX = 0;

    const qNearArm = new THREE.Quaternion();
    const qNearFore = new THREE.Quaternion();
    const qNearHand = new THREE.Quaternion();
    const qRestArm = new THREE.Quaternion();
    const qRestFore = new THREE.Quaternion();
    const qRestHand = new THREE.Quaternion();

    const mousePos = { x: 0, y: 0 };
    const onMouseMove = e => {
      const r = container.getBoundingClientRect();
      mousePos.x = THREE.MathUtils.clamp(((e.clientX - r.left) / r.width) * 2 - 1, -1, 1);
      mousePos.y = THREE.MathUtils.clamp(-(((e.clientY - r.top) / r.height) * 2 - 1), -1, 1);
    };
    window.addEventListener('mousemove', onMouseMove);

    try {
      scene = new THREE.Scene();
      const W = container.clientWidth  || 400;
      const H = container.clientHeight || 560;

      // Tight half-body framing ends around the upper waist.
      camera = new THREE.PerspectiveCamera(23, W / H, 0.1, 20);
      camera.position.set(0, 1.45, 2.00);
      camera.lookAt(0, 1.45, 0);

      try {
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'default' });
      } catch (err1) {
        console.warn('Default WebGL init failed, falling back to basic context:', err1);
        renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
      }

      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(W, H);
      renderer.toneMapping         = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.12;
      renderer.outputColorSpace    = THREE.SRGBColorSpace;
      container.appendChild(renderer.domElement);

      const handleContextLost = (e) => {
        e.preventDefault();
        console.warn('[Nova3D] WebGL context lost');
      };
      renderer.domElement.addEventListener('webglcontextlost', handleContextLost, false);

      scene.add(new THREE.HemisphereLight(0xf3f6ff, 0x8d7a66, 1.15));
      const key = new THREE.DirectionalLight(0xfff4e8, 2.05);
      key.position.set(1.15, 2.4, 2.2);
      scene.add(key);
      const fill = new THREE.DirectionalLight(0xc5d4ee, 0.95);
      fill.position.set(-1.6, 1.5, 1.4);
      scene.add(fill);
      const rim = new THREE.DirectionalLight(0xe8f0ff, 1.35);
      rim.position.set(-0.2, 1.9, -2.0);
      scene.add(rim);
      scene.add(new THREE.AmbientLight(0xffffff, 0.55));

      function setupModel(gltf) {
        if (isDisposed) return;
        const model = skeletonClone(gltf.scene);
        modelRoot = model;

        model.traverse(child => {
          if (child.isBone) {
            const n = child.name;
            if (n === 'Head')        headBone        = child;
            if (n === 'Neck')        neckBone        = child;
            if (n === 'Spine2')      spineBone       = child;
            if (n === 'Spine1')      spine1Bone      = child;
            if (n === 'RightArm')     rightArmBone     = child;
            if (n === 'RightForeArm') rightForeArmBone = child;
            if (n === 'RightHand')    rightHandBone    = child;
            if (n === 'LeftArm')      leftArmBone      = child;
            if (n === 'LeftForeArm')  leftForeArmBone  = child;
            if (n === 'LeftHand')     leftHandBone     = child;
            if (/^(Left|Right)Hand(Thumb|Index|Middle|Ring|Pinky)[123]$/.test(n)) {
              fingerBones.push({
                bone: child,
                rest: child.quaternion.clone(),
                curl: curlForFinger(n),
              });
            }
          }
          if (child.isMesh) {
            child.frustumCulled = false;
            if (/^Eye(Left|Right)$/.test(child.name)) eyeObjects.push(child);
            if (child.morphTargetDictionary && child.morphTargetInfluences) {
              morphMeshes.push(child);
            }
            if (child.material) styleMaterial(child);
          }
        });

        if (rightArmBone)     restRightArm     = rightArmBone.quaternion.clone();
        if (rightForeArmBone) restRightForeArm = rightForeArmBone.quaternion.clone();
        if (rightHandBone)    restRightHand    = rightHandBone.quaternion.clone();
        if (leftArmBone)      restLeftArm      = leftArmBone.quaternion.clone();
        if (leftForeArmBone)  restLeftForeArm  = leftForeArmBone.quaternion.clone();
        if (leftHandBone)     restLeftHand     = leftHandBone.quaternion.clone();
        if (headBone) {
          restHeadRotX = headBone.rotation.x;
          restHeadRotY = headBone.rotation.y;
          restHeadRotZ = headBone.rotation.z;
        }
        if (neckBone) {
          restNeckRotX = neckBone.rotation.x;
          restNeckRotY = neckBone.rotation.y;
        }
        if (spineBone)  restSpineRotX  = spineBone.rotation.x;
        if (spine1Bone) restSpine1RotX = spine1Bone.rotation.x;

        // Keep both arms close to the body in the professional rest pose.
        if (restRightArm) {
          qNearArm.copy(restRightArm).multiply(
            new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.26, 0.58, 0.88))
          );
        }
        if (restRightForeArm) {
          qNearFore.copy(restRightForeArm).multiply(
            new THREE.Quaternion().setFromEuler(new THREE.Euler(0.78, -0.48, 0.22))
          );
        }
        if (restRightHand) {
          qNearHand.copy(restRightHand).multiply(
            new THREE.Quaternion().setFromEuler(new THREE.Euler(0.04, -0.08, 0.18))
          );
        }
        if (restLeftArm) {
          qRestArm.copy(restLeftArm).multiply(
            new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.26, -0.58, -0.88))
          );
        }
        if (restLeftForeArm) {
          qRestFore.copy(restLeftForeArm).multiply(
            new THREE.Quaternion().setFromEuler(new THREE.Euler(0.78, 0.48, -0.22))
          );
        }
        if (restLeftHand) {
          qRestHand.copy(restLeftHand).multiply(
            new THREE.Quaternion().setFromEuler(new THREE.Euler(0.04, 0.08, -0.18))
          );
        }
        scene.add(model);
        setLoading(false);
      }

      if (_cachedGLTF) {
        setupModel(_cachedGLTF);
      } else {
        preloadNova()
          .then(setupModel)
          .catch(err => {
            console.error('Error loading 3D avatar:', err);
            if (!isDisposed) { setLoadError('Failed to load 3D character.'); setLoading(false); }
          });
      }

      const setMorphWeight = (name, w) => {
        for (const mesh of morphMeshes) {
          const idx = mesh.morphTargetDictionary[name];
          if (idx !== undefined) mesh.morphTargetInfluences[idx] = w;
        }
      };

      const clock = new THREE.Clock();
      let nextBlinkTime = 2, blinkProgress = 0, isBlinking = false;
      let visemeTimer = 0, currentViseme = '';
      let targetVisemeWeight = 0, currentVisemeWeight = 0;
      let currentJawWeight = 0, targetJawWeight = 0;
      let speechEnvelope = 0;
      let smileWeight = 0.2;
      const audioSamples = new Uint8Array(128);
      const headDelta = { x: 0, y: 0, z: 0 };
      const neckDelta = { x: 0, y: 0 };
      const eyeDelta = { x: 0, y: 0 };
      const qTmp = new THREE.Quaternion();
      const qCurl = new THREE.Quaternion();

      const animate = () => {
        if (isDisposed) return;
        animationFrameId = requestAnimationFrame(animate);

        const delta   = Math.min(clock.getDelta(), 0.1);
        const elapsed = clock.getElapsedTime();
        const cst     = statusRef.current;
        const thinking = cst === 'processing';
        const eyeLookX = mousePos.x * 0.065 + Math.sin(elapsed * 0.9) * 0.008;
        const eyeLookY = -mousePos.y * 0.045 + Math.sin(elapsed * 1.15) * 0.006;

        setMorphWeight('eyeWideLeft', 0);
        setMorphWeight('eyeWideRight', 0);
        setMorphWeight('eyeSquintLeft', 0);
        setMorphWeight('eyeSquintRight', 0);
        setMorphWeight('cheekSquintLeft', 0);
        setMorphWeight('cheekSquintRight', 0);

        if (elapsed > nextBlinkTime) {
          isBlinking = true;
          blinkProgress += delta * 12;
          if (blinkProgress >= 1) {
            blinkProgress = 0; isBlinking = false;
            nextBlinkTime = elapsed + 2.5 + Math.random() * 3.5;
          }
        }
        const bv = isBlinking ? Math.sin(blinkProgress * Math.PI) : 0;
        setMorphWeight('eyeBlinkLeft', bv); setMorphWeight('eyeBlinkRight', bv);

        if (cst === 'speaking') {
          let audioLevel = 0;
          const analyser = window.__novaTtsAnalyser;
          if (window.__novaTtsActive && analyser) {
            // Frequency-band energy in the speech range (100–4000 Hz)
            const freqData = new Uint8Array(analyser.frequencyBinCount);
            analyser.getByteFrequencyData(freqData);
            const sampleRate = analyser.context.sampleRate;
            const binHz = sampleRate / analyser.fftSize;
            const loIdx = Math.max(1, Math.round(100 / binHz));
            const hiIdx = Math.min(freqData.length - 1, Math.round(4000 / binHz));
            let sum = 0, count = 0;
            for (let k = loIdx; k <= hiIdx; k++) { sum += freqData[k]; count++; }
            const avgFreq = count > 0 ? sum / count : 0;
            // Softer normalisation — keeps mouth in human range even for loud TTS
            audioLevel = THREE.MathUtils.clamp((avgFreq - 8) / 140, 0, 1);
          }
          // Fast attack, slower decay — matches how lips open/close with syllables
          const envelopeRate = audioLevel > speechEnvelope ? 35 : 14;
          speechEnvelope = THREE.MathUtils.lerp(speechEnvelope, audioLevel, delta * envelopeRate);
          const voiced = speechEnvelope > 0.05;

          // ── Syllable-rhythm jaw ────────────────────────────────────────────
          // Rather than constant-open, the jaw pulses open/closed at a rate
          // that tracks speech energy (~4–7 syllables/sec). At rest between
          // syllables the jaw naturally returns toward closed.
          visemeTimer -= delta;
          if (visemeTimer <= 0) {
            // 80–180 ms per syllable group — faster when more energetic speech
            const sylRate = 0.08 + (1 - speechEnvelope) * 0.10;
            visemeTimer = sylRate + Math.random() * 0.04;
            if (currentViseme) setMorphWeight(currentViseme, 0);
            // Lip-shape visemes at low weight — subtle, not dominant
            currentViseme = VISEME_SEQUENCE[Math.floor(Math.random() * VISEME_SEQUENCE.length)];
            targetVisemeWeight = voiced ? 0.03 + speechEnvelope * 0.07 : 0;
          }
          if (!voiced) targetVisemeWeight = 0;
          currentVisemeWeight = THREE.MathUtils.lerp(currentVisemeWeight, targetVisemeWeight, delta * 28);

          // Jaw: resting ~0, normal speech ~0.20–0.30, stressed vowel up to 0.38
          const jawPulse = voiced ? Math.max(0, Math.sin(elapsed * (5 + speechEnvelope * 3)) * 0.5 + 0.5) : 0;
          targetJawWeight  = voiced ? (0.05 + speechEnvelope * 0.26) * (0.55 + jawPulse * 0.45) : 0;
          currentJawWeight = THREE.MathUtils.lerp(currentJawWeight, targetJawWeight, delta * (voiced ? 28 : 22));

          if (currentViseme) setMorphWeight(currentViseme, currentVisemeWeight);
          // Cap at 0.38 — visible, natural, not cartoon-wide
          const clampedJaw = Math.min(currentJawWeight, 0.38);
          setMorphWeight('jawOpen', clampedJaw);
          setMorphWeight('mouthOpen', clampedJaw * 0.22);
          smileWeight = THREE.MathUtils.lerp(smileWeight, 0.22, delta * 3);
          setMorphWeight('mouthSmile', smileWeight);
          setMorphWeight('browInnerUp', 0.10 + Math.sin(elapsed * 3.5) * 0.06);
        } else {
          if (currentViseme) { setMorphWeight(currentViseme, 0); currentViseme = ''; }
          currentVisemeWeight = THREE.MathUtils.lerp(currentVisemeWeight, 0, delta * 15);
          currentJawWeight    = THREE.MathUtils.lerp(currentJawWeight, 0, delta * 15);
          speechEnvelope       = THREE.MathUtils.lerp(speechEnvelope, 0, delta * 12);
          setMorphWeight('jawOpen', currentJawWeight);
          setMorphWeight('mouthOpen', currentJawWeight);
          const idleSmile = (cst === 'idle' || cst === 'ready') ? 0.28 : cst === 'listening' ? 0.16 : 0.06;
          smileWeight = THREE.MathUtils.lerp(smileWeight, idleSmile, delta * 5);
          setMorphWeight('mouthSmile', smileWeight);
        }

        let tHX = 0, tHY = 0, tHZ = 0, tBrow = 0;
        const gx = mousePos.x * 0.09, gy = -mousePos.y * 0.06;

        if (cst === 'speaking') {
          tHX = Math.sin(elapsed * 4.2) * 0.03 - 0.03 + gy * 0.5;
          tHY = Math.sin(elapsed * 1.8) * 0.03 + gx * 0.5;
          tHZ = Math.sin(elapsed * 2.1) * 0.015; tBrow = 0.15;
        } else if (cst === 'listening') {
          tHX = -0.02 + gy * 0.7; tHY = gx * 0.8; tHZ = 0.03; tBrow = 0.22;
          setMorphWeight('eyeWideLeft', 0.15); setMorphWeight('eyeWideRight', 0.15);
        } else if (cst === 'happy' || cst === 'delighted') {
          tHX = gy * 0.55; tHY = gx * 0.75; tHZ = -0.015; tBrow = 0.22;
          setMorphWeight('eyeSquintLeft', 0.16); setMorphWeight('eyeSquintRight', 0.16);
          setMorphWeight('cheekSquintLeft', 0.16); setMorphWeight('cheekSquintRight', 0.16);
          smileWeight = THREE.MathUtils.lerp(smileWeight, 0.5, delta * 6);
        } else if (cst === 'curious') {
          tHX = -0.03 + gy * 0.65; tHY = gx * 0.8; tHZ = 0.12; tBrow = 0.3;
          setMorphWeight('eyeWideLeft', 0.12); setMorphWeight('eyeWideRight', 0.12);
        } else if (cst === 'confused') {
          tHX = gy * 0.5; tHY = gx * 0.65; tHZ = Math.sin(elapsed * 2.2) * 0.08; tBrow = 0.12;
          setMorphWeight('browDownLeft', 0.1); setMorphWeight('browDownRight', 0.2);
          setMorphWeight('eyeWideLeft', 0.12);
        } else if (cst === 'enthusiastic') {
          tHX = gy * 0.6; tHY = gx * 0.85; tHZ = Math.sin(elapsed * 2) * 0.015; tBrow = 0.34;
          setMorphWeight('eyeWideLeft', 0.2); setMorphWeight('eyeWideRight', 0.2);
          smileWeight = THREE.MathUtils.lerp(smileWeight, 0.42, delta * 6);
        } else if (thinking) {
          tHX = 0.02 + Math.sin(elapsed * 1.6) * 0.008;
          tHY = -0.12;
          tHZ = 0.06 + Math.sin(elapsed * 2.2) * 0.01;
          tBrow = 0.38;
          setMorphWeight('browDownLeft', 0.18);
          setMorphWeight('browDownRight', 0.08);
          setMorphWeight('eyeWideLeft', 0);
          setMorphWeight('eyeWideRight', 0);
        } else {
          tHX = Math.sin(elapsed * 0.8) * 0.015 - 0.03 + gy * 0.6;
          tHY = Math.sin(elapsed * 0.5) * 0.025 + gx * 0.6;
          tHZ = Math.sin(elapsed * 0.4) * 0.01; tBrow = 0.05;
          setMorphWeight('eyeWideLeft', 0); setMorphWeight('eyeWideRight', 0);
          setMorphWeight('browDownLeft', 0); setMorphWeight('browDownRight', 0);
        }
        setMorphWeight('browInnerUp', tBrow);

        headDelta.x = THREE.MathUtils.lerp(headDelta.x, tHX, delta * 6);
        headDelta.y = THREE.MathUtils.lerp(headDelta.y, tHY, delta * 6);
        headDelta.z = THREE.MathUtils.lerp(headDelta.z, tHZ, delta * 6);
        neckDelta.x = THREE.MathUtils.lerp(neckDelta.x, tHX * 0.4, delta * 4);
        neckDelta.y = THREE.MathUtils.lerp(neckDelta.y, tHY * 0.4, delta * 4);

        if (headBone) {
          headBone.rotation.x = restHeadRotX + headDelta.x;
          headBone.rotation.y = restHeadRotY + headDelta.y;
          headBone.rotation.z = restHeadRotZ + headDelta.z;
        }
        if (neckBone) {
          neckBone.rotation.x = restNeckRotX + neckDelta.x;
          neckBone.rotation.y = restNeckRotY + neckDelta.y;
        }
        eyeDelta.x = THREE.MathUtils.lerp(eyeDelta.x, eyeLookY, delta * 9);
        eyeDelta.y = THREE.MathUtils.lerp(eyeDelta.y, eyeLookX, delta * 9);
        for (const eye of eyeObjects) {
          eye.rotation.x = eyeDelta.x;
          eye.rotation.y = eyeDelta.y;
        }
        const breath = Math.sin(elapsed * 1.35) * 0.014;
        if (spineBone)  spineBone.rotation.x  = restSpineRotX  + breath;
        if (spine1Bone) spine1Bone.rotation.x = restSpine1RotX + breath * 0.6;
        if (modelRoot) {
          modelRoot.rotation.y = Math.sin(elapsed * 0.35) * 0.03;
        }

        if (restRightArm && rightArmBone) {
          qTmp.copy(restRightArm).slerp(qNearArm, 1);
          rightArmBone.quaternion.copy(qTmp);
        }
        if (restRightForeArm && rightForeArmBone) {
          qTmp.copy(restRightForeArm).slerp(qNearFore, 1);
          rightForeArmBone.quaternion.copy(qTmp);
        }
        if (restRightHand && rightHandBone) {
          qTmp.copy(qNearHand);
          qCurl.setFromEuler(new THREE.Euler(0, 0, 0));
          qTmp.multiply(qCurl);
          rightHandBone.quaternion.copy(qTmp);
        }
        if (restLeftArm && leftArmBone) {
          leftArmBone.quaternion.copy(qRestArm);
        }
        if (restLeftForeArm && leftForeArmBone) {
          leftForeArmBone.quaternion.copy(qRestFore);
        }
        if (restLeftHand && leftHandBone) {
          leftHandBone.quaternion.copy(qRestHand);
        }
        for (const f of fingerBones) {
          qCurl.setFromEuler(f.curl);
          qTmp.copy(f.rest).multiply(qCurl);
          f.bone.quaternion.copy(f.rest).slerp(qTmp, 0.35);
        }

        renderer.render(scene, camera);
      };
      animate();

      const ro = new ResizeObserver(entries => {
        if (!entries?.length) return;
        const { width: w, height: h } = entries[0].contentRect;
        if (w > 0 && h > 0 && camera && renderer) {
          camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h);
        }
      });
      ro.observe(container);

      return () => {
        isDisposed = true;
        cancelAnimationFrame(animationFrameId);
        window.removeEventListener('mousemove', onMouseMove);
        ro.disconnect();
        if (renderer?.domElement) {
          renderer.domElement.removeEventListener('webglcontextlost', handleContextLost);
        }
        if (renderer) {
          try {
            renderer.forceContextLoss?.();
            renderer.dispose();
          } catch (_) {}
        }
        if (renderer?.domElement && container.contains(renderer.domElement)) {
          container.removeChild(renderer.domElement);
        }
        scene?.traverse(obj => {
          if (obj.geometry) obj.geometry.dispose();
          if (obj.material) (Array.isArray(obj.material) ? obj.material : [obj.material]).forEach(m => m.dispose());
        });
      };
    } catch (e) {
      console.error('Three.js setup error:', e);
      setLoadError('Failed to initialize 3D rendering context.');
      setLoading(false);
    }
  }, []);

  return (
    <div style={{
      position: 'relative',
      width: size.width || '100%',
      height: size.height || '560px',
      maxWidth: '460px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      overflow: 'hidden',
    }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative', zIndex: 2, cursor: 'grab' }} />

      {st === 'processing' && (
        <div className="nova-thinking-bubble" aria-label="Nova is thinking">
          <span className="nova-thinking-dot" />
          <span className="nova-thinking-dot" />
          <span className="nova-thinking-dot" />
        </div>
      )}

      {loading && !loadError && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '12px', zIndex: 3, background: 'radial-gradient(circle, rgba(255,255,255,0.7) 0%, rgba(240,244,255,0.3) 100%)', backdropFilter: 'blur(4px)', borderRadius: '24px' }}>
          <div style={{ width: '42px', height: '42px', border: '3.5px solid rgba(26,35,126,0.15)', borderTopColor: '#1a237e', borderRadius: '50%', animation: 'spin3d 0.8s linear infinite' }} />
          <span style={{ fontSize: '13px', fontWeight: '700', color: '#1a237e', letterSpacing: '0.4px' }}>Loading Nova…</span>
          <style>{`@keyframes spin3d { to { transform: rotate(360deg); } }`}</style>
        </div>
      )}

      {loadError && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '16px', textAlign: 'center', zIndex: 3 }}>
          <div style={{ width: '120px', height: '120px', borderRadius: '50%', background: 'linear-gradient(135deg, #1a237e 0%, #3949ab 50%, #00e5ff 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 10px 28px rgba(26,35,126,0.25)', marginBottom: '12px' }}>
            <span style={{ fontSize: '50px' }}>👩🏻‍💼</span>
          </div>
          <span style={{ fontSize: '15px', fontWeight: '700', color: '#1a237e' }}>Nova</span>
          <span style={{ fontSize: '12px', color: '#5c6bc0', marginTop: '2px' }}>Digital Receptionist</span>
        </div>
      )}

      <div style={{
        position: 'absolute', width: '280px', height: '340px', borderRadius: '50%',
        background: st === 'speaking'
          ? 'radial-gradient(circle, rgba(124,77,255,0.30) 0%, transparent 70%)'
          : st === 'listening'
            ? 'radial-gradient(circle, rgba(67,160,71,0.30) 0%, transparent 70%)'
            : st === 'processing'
              ? 'radial-gradient(circle, rgba(255,179,0,0.32) 0%, transparent 70%)'
              : 'radial-gradient(circle, rgba(26,35,126,0.20) 0%, transparent 70%)',
        bottom: '36px', left: '50%', transform: 'translateX(-50%)',
        pointerEvents: 'none', transition: 'all 0.6s ease', zIndex: 1,
        animation: st === 'speaking' ? 'pulseHalo 1.2s ease-in-out infinite' : 'pulseHalo 4s ease-in-out infinite',
      }} />

      <div style={{
        position: 'absolute', bottom: '8px', width: '210px', height: '22px',
        borderRadius: '50%', background: 'radial-gradient(ellipse at center, rgba(0,0,0,0.20) 0%, transparent 75%)',
        pointerEvents: 'none', zIndex: 1,
      }} />

      <style>{`
        @keyframes pulseHalo {
          0%, 100% { transform: translateX(-50%) scale(1);   opacity: 0.75; }
          50%       { transform: translateX(-50%) scale(1.08); opacity: 1; }
        }
        .nova-thinking-bubble {
          position: absolute;
          top: 8%;
          right: 15%;
          display: flex;
          align-items: center;
          gap: 7px;
          padding: 13px 15px;
          border: 1px solid rgba(255, 179, 0, 0.35);
          border-radius: 18px 18px 5px 18px;
          background: rgba(255, 251, 235, 0.92);
          box-shadow: 0 10px 24px rgba(92, 67, 20, 0.15);
          z-index: 4;
          animation: novaBubbleFloat 2.4s ease-in-out infinite;
        }
        .nova-thinking-dot {
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background: #d88900;
          animation: novaDotBounce 1.15s ease-in-out infinite;
        }
        .nova-thinking-dot:nth-child(2) { animation-delay: 0.15s; }
        .nova-thinking-dot:nth-child(3) { animation-delay: 0.3s; }
        @keyframes novaBubbleFloat {
          0%, 100% { transform: translateY(0) rotate(2deg); }
          50% { transform: translateY(-7px) rotate(-2deg); }
        }
        @keyframes novaDotBounce {
          0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
          30% { transform: translateY(-5px); opacity: 1; }
        }
        @media (max-width: 560px) {
          .nova-thinking-bubble { top: 7%; right: 10%; transform: scale(0.88); }
        }
      `}</style>
    </div>
  );
}

function curlForFinger(name) {
  const joint = Number(name.slice(-1));
  const base = /Index/.test(name) ? 0.18
    : /Thumb/.test(name) ? 0.22
    : /Middle/.test(name) ? 0.55
    : /Ring/.test(name) ? 0.72
    : 0.82;
  const x = base * (joint === 1 ? 1 : joint === 2 ? 0.85 : 0.55);
  if (/Thumb/.test(name)) return new THREE.Euler(x * 0.4, 0.15, 0.35);
  return new THREE.Euler(x, 0, 0);
}
