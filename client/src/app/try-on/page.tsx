"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { submitFeedback } from "../../../api/api";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import * as THREE from "three";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

// ─── Landmark indices (MediaPipe 478-point model) ────────────────────────────
const LM = {
  LEFT_EYE_OUTER: 33,
  RIGHT_EYE_OUTER: 263,
  NOSE_BRIDGE: 168,
  LEFT_TEMPLE: 234,
  RIGHT_TEMPLE: 454,
} as const;

function toWorld(
  mappedX: number,
  mappedY: number,
  camera: THREE.OrthographicCamera,
): THREE.Vector3 {
  const x = (mappedX - 0.5) * (camera.right - camera.left);
  const y = -(mappedY - 0.5) * (camera.top - camera.bottom);
  return new THREE.Vector3(x, y, 0);
}

// ─── Component ───────────────────────────────────────────────────────────────
export default function TryOnPage() {
  const router = useRouter();
  const [data, setData] = useState<any>(null);
  const [activeGlass, setActiveGlass] = useState<any>(null);
  const [votedGlasses, setVotedGlasses] = useState<number[]>([]);
  const [status, setStatus] = useState<string>("Starting webcam…");
  const [modelLoading, setModelLoading] = useState(false);
  const [webcamError, setWebcamError] = useState<string | null>(null);
  const [sceneReady, setSceneReady] = useState(false);

  // DOM refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Three.js & MediaPipe refs
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.OrthographicCamera | null>(null);
  const glassesGroupRef = useRef<THREE.Group | null>(null);
  const loaderRef = useRef(new OBJLoader());
  
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const rafRef = useRef<number>(0);
  const lastVideoTimeRef = useRef(-1);
  const streamRef = useRef<MediaStream | null>(null);

  // ── 1. Load session data ──────────────────────────────────────────────────
  useEffect(() => {
    const storedRes = sessionStorage.getItem("ai_recommendations");
    if (storedRes) {
      const parsed = JSON.parse(storedRes);
      setData(parsed);
      setActiveGlass(parsed.top_matches[0]);
    } else {
      router.push("/");
    }
  }, [router]);

  // ── 2. Bootstrap: init Three.js + webcam + MediaPipe, then start loop ─────
  useEffect(() => {
    // We must wait for data to load so the early return finishes and the canvas is mounted
    if (!data) return;

    const canvas = overlayCanvasRef.current;
    if (!canvas) return;

    // Use setTimeout so the DOM has completely painted its initial flex layout
    const timeoutId = setTimeout(() => {
      // ── Init Three.js ───────────────────────────────────────────────────
      const cW = canvas.clientWidth || 640;
      const cH = canvas.clientHeight || 480;

      const renderer = new THREE.WebGLRenderer({
        canvas,
        alpha: true,
        antialias: true,
      });
      renderer.setPixelRatio(window.devicePixelRatio);
      // Pass false to avoid overriding the CSS w-full h-full dimensions
      renderer.setSize(cW, cH, false);
      renderer.setClearColor(0x000000, 0);

      const scene = new THREE.Scene();

      const aspect = cW / cH;
      const halfH = 5;
      const camera = new THREE.OrthographicCamera(
        -halfH * aspect,
        halfH * aspect,
        halfH,
        -halfH,
        0.01,
        100,
      );
      camera.position.set(0, 0, 10);
      camera.lookAt(0, 0, 0);

      // Enhance lighting so materials remain visible regardless of color
      scene.add(new THREE.AmbientLight(0xffffff, 2.5));
      const dirLight = new THREE.DirectionalLight(0xffffff, 3.0);
      dirLight.position.set(0, 5, 10);
      scene.add(dirLight);
      const fillLight = new THREE.DirectionalLight(0xeef4ff, 1.5);
      fillLight.position.set(-5, -2, 8);
      scene.add(fillLight);

      rendererRef.current = renderer;
      sceneRef.current = scene;
      cameraRef.current = camera;
      setSceneReady(true); // Now we can safely load models
      
      // ── Start rendering loop ────────────────────────────────────────────
      const loop = () => {
        rafRef.current = requestAnimationFrame(loop);

        const video = videoRef.current;
        const rendererDOM = renderer.domElement;
        if (!video || video.readyState < 2 || !rendererDOM) return;

        const nowMs = performance.now();
        const lmk = landmarkerRef.current;
        const glasses = glassesGroupRef.current;

        if (lmk && video.currentTime !== lastVideoTimeRef.current) {
          lastVideoTimeRef.current = video.currentTime;
          const result = lmk.detectForVideo(video, nowMs);

          if (result.faceLandmarks?.length && glasses) {
            const marks = result.faceLandmarks[0];

            // 1. Math to correct the object-cover video crop
            // MediaPipe gives [0,1] coord based on the RAW video source.
            // Our video element is forced to object-cover the container.
            const wC = rendererDOM.clientWidth;
            const hC = rendererDOM.clientHeight;
            const vW = video.videoWidth || 1280;
            const vH = video.videoHeight || 720;

            const vAspect = vW / vH;
            const cAspect = wC / hC;
            let renderedW, renderedH;
            
            if (cAspect > vAspect) {
              renderedW = wC;
              renderedH = wC / vAspect;
            } else {
              renderedH = hC;
              renderedW = hC * vAspect;
            }
            
            const xOffset = (renderedW - wC) / 2;
            const yOffset = (renderedH - hC) / 2;

            // mapC maps raw MediaPipe coords to Canvas coords correctly
            const mapC = (raw: any) => ({
              x: (raw.x * renderedW - xOffset) / wC,
              y: (raw.y * renderedH - yOffset) / hC,
              z: raw.z, // retain native depth metric
            });

            const leftEyeRaw = marks[LM.LEFT_EYE_OUTER];
            const rightEyeRaw = marks[LM.RIGHT_EYE_OUTER];
            const noseRaw = marks[LM.NOSE_BRIDGE];
            const leftTempleRaw = marks[LM.LEFT_TEMPLE];
            const rightTempleRaw = marks[LM.RIGHT_TEMPLE];

            const leftEyeScreen = mapC(leftEyeRaw);
            const rightEyeScreen = mapC(rightEyeRaw);
            const noseScreen = mapC(noseRaw);
            const leftTempleScreen = mapC(leftTempleRaw);
            const rightTempleScreen = mapC(rightTempleRaw);

            const leftEyeW = toWorld(leftEyeScreen.x, leftEyeScreen.y, camera);
            const rightEyeW = toWorld(rightEyeScreen.x, rightEyeScreen.y, camera);
            const noseW = toWorld(noseScreen.x, noseScreen.y, camera);
            const leftTempleW = toWorld(leftTempleScreen.x, leftTempleScreen.y, camera);
            const rightTempleW = toWorld(rightTempleScreen.x, rightTempleScreen.y, camera);

            // Compute face transforms
            const eyeMidX = (leftEyeW.x + rightEyeW.x) / 2;
            const eyeMidY = (leftEyeW.y + rightEyeW.y) / 2;
            const faceWidth = leftTempleW.distanceTo(rightTempleW);

            // Roll: math.atan2(y, x). User's physical right eye is on the left side of unmirrored screen
            // so leftEyeW.x > rightEyeW.x, generating a default roll of 0!
            const roll = Math.atan2(
              leftEyeW.y - rightEyeW.y,
              leftEyeW.x - rightEyeW.x,
            );
            
            // Pitch: rotate up/down depending on nose depth. 
            const pitch = (noseRaw.z - (leftEyeRaw.z + rightEyeRaw.z) * 0.5) * 2.5;
            
            // Yaw: rotate left/right
            const faceCenterRawX = (leftTempleRaw.x + rightTempleRaw.x) / 2;
            const yaw = -(noseRaw.x - faceCenterRawX) * Math.PI * 1.5;

            // Rest glasses slightly above nose
            const eyeToNoseY = noseW.y - eyeMidY;
            const glassesY = eyeMidY + eyeToNoseY * 0.15;

            glasses.position.set(eyeMidX, glassesY, 0);
            glasses.scale.setScalar(faceWidth * 0.95); // Adjust multiplier tighter to fit exact face width
            glasses.rotation.set(pitch, yaw, roll);
            glasses.visible = true;
          } else if (glasses) {
            glasses.visible = false;
          }
        }

        renderer.render(scene, camera);
      };

      // ── Start webcam ──────────────────────────────────────────────────
      const startCam = async () => {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
          });
          streamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            await new Promise<void>((res) => {
              videoRef.current!.onloadedmetadata = () => res();
            });
            await videoRef.current.play();
          }
          setWebcamError(null);
        } catch (e) {
          console.error("Webcam error:", e);
          setWebcamError("Camera access denied. Please allow camera access and reload.");
        }
      };

      // ── Init MediaPipe ────────────────────────────────────────────────
      const startMediaPipe = async () => {
        setStatus("Loading AI tracker…");
        try {
          const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
          );
          const landmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
              delegate: "GPU",
            },
            runningMode: "VIDEO",
            numFaces: 1,
            outputFacialTransformationMatrixes: true,
          });
          landmarkerRef.current = landmarker;
          setStatus("Ready");
        } catch (e) {
          console.error("Failed to load MediaPipe", e);
          setStatus("Error loading tracker");
        }
      };

      // Ensure we immediately start mapping
      rafRef.current = requestAnimationFrame(loop);
      startCam();
      startMediaPipe();

      // ── Handle resize ─────────────────────────────────────────────────
      const handleResize = () => {
        const cw = canvas.clientWidth;
        const ch = canvas.clientHeight;
        if (!cw || !ch) return;
        renderer.setSize(cw, ch, false);
        const a = cw / ch;
        camera.left = -halfH * a;
        camera.right = halfH * a;
        camera.updateProjectionMatrix();
      };
      window.addEventListener("resize", handleResize);

      // Link cleanup
      (canvas as any).__cleanup = () => {
        cancelAnimationFrame(rafRef.current);
        window.removeEventListener("resize", handleResize);
        streamRef.current?.getTracks().forEach((t) => t.stop());
        renderer.dispose();
        landmarkerRef.current?.close();
      };
    }, 50);

    return () => {
      clearTimeout(timeoutId);
      const canvas = overlayCanvasRef.current;
      if (canvas && (canvas as any).__cleanup) {
        (canvas as any).__cleanup();
      }
    };
  }, [data]);

  // ── 3. Load / swap glasses model (gated by sceneReady) ────────────────────
  useEffect(() => {
    if (!sceneReady || !sceneRef.current || !activeGlass) return;

    if (glassesGroupRef.current) {
      sceneRef.current.remove(glassesGroupRef.current);
      glassesGroupRef.current = null;
    }

    setModelLoading(true);

    // Resolves model name seamlessly
    const fileName =
      activeGlass.file_name ||
      `glass_${String(activeGlass.glass_id + 1).padStart(2, "0")}`;
    const modelPath = `/models/${fileName}.obj`;

    console.log("Loading model:", modelPath);

    loaderRef.current.load(
      modelPath,
      (obj) => {
        // We use nested groups to perfectly center and scale the model
        const mainGroup = new THREE.Group();
        const scaleWrapper = new THREE.Group();

        // Apply a realistic material fallback since OBJ files don't have materials included natively here
        const defaultMat = new THREE.MeshStandardMaterial({
          color: 0x2a2a2a,
          roughness: 0.3,
          metalness: 0.7,
        });
        obj.traverse((child: any) => {
          if (child.isMesh) {
            child.material = defaultMat;
            child.castShadow = true;
            child.receiveShadow = true;
          }
        });

        // 1. Center the model exactly.
        const box = new THREE.Box3().setFromObject(obj);
        const center = box.getCenter(new THREE.Vector3());
        
        // Set the inverse center offset at the parent 'obj' transform
        obj.position.set(-center.x, -center.y, -center.z);
        
        // Setup orientation holder because these models are exported upside down (Y-axis inverted constraint)
        const objHolder = new THREE.Group();
        objHolder.add(obj);
        objHolder.rotation.z = Math.PI;

        // Add it to unscaled wrapper, which makes the wrapper's local (0,0,0) the center of the glasses
        scaleWrapper.add(objHolder);

        // 2. Normalize the scale wrapper bounding box to exactly 1.0 unit width
        // By normalizing based purely on size.x (the width of the glasses), we ensure the glasses frame scales 
        // horizontally to exactly match the tracked face width, ignoring variations in stem depth or Y-height!
        const size = box.getSize(new THREE.Vector3());
        if (size.x > 0) {
          scaleWrapper.scale.setScalar(1 / size.x);
        }

        // 3. Mount fully centered wrapper to the group controlled by the Try-On face tracker loop.
        mainGroup.add(scaleWrapper);
        mainGroup.visible = false;
        
        sceneRef.current!.add(mainGroup);
        glassesGroupRef.current = mainGroup;
        setModelLoading(false);
        console.log("Model loaded successfully:", modelPath);
      },
      (progress) => {
        if (progress.total) {
          console.log(`Loading ${fileName}: ${Math.round((progress.loaded / progress.total) * 100)}%`);
        }
      },
      (err) => {
        console.error("OBJ load error for:", modelPath, err);
        setModelLoading(false);
      },
    );
  }, [activeGlass, sceneReady]);

  // ── Retry webcam handler ──────────────────────────────────────────────────
  const retryWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await new Promise<void>((res) => {
          videoRef.current!.onloadedmetadata = () => res();
        });
        await videoRef.current.play();
      }
      setWebcamError(null);
    } catch (e) {
      console.error("Webcam error:", e);
      setWebcamError("Camera access denied. Please allow camera access and reload.");
    }
  }, []);

  // ── Feedback handler ──────────────────────────────────────────────────────
  const handleFeedback = async (liked: boolean) => {
    if (!data || !activeGlass) return;
    if (votedGlasses.includes(activeGlass.glass_id)) {
      alert("You've already voted on these glasses!");
      return;
    }
    const featuresDict = JSON.parse(sessionStorage.getItem("user_features") || "{}");
    const featuresArray = [
      featuresDict.cheek_jaw_ratio || 1.0,
      featuresDict.face_hw_ratio || 1.0,
      featuresDict.midface_ratio || 1.0,
    ];
    try {
      await submitFeedback({
        glass_id: activeGlass.glass_id,
        detected_face_shape: data.detected_face_shape_id,
        features: featuresArray,
        liked,
      });
      setVotedGlasses((prev) => [...prev, activeGlass.glass_id]);
      alert(`Feedback saved! You ${liked ? "liked" : "disliked"} these glasses.`);
    } catch (err) {
      console.error(err);
      alert("Error saving feedback.");
    }
  };

  if (!data) {
    return (
      <div className="flex items-center justify-center h-screen bg-[#0a0a0f] text-white">
        <div className="text-center space-y-3">
          <div className="w-10 h-10 border-2 border-white/20 border-t-white rounded-full animate-spin mx-auto" />
          <p className="text-sm tracking-widest uppercase text-white/50">Loading AI matches…</p>
        </div>
      </div>
    );
  }

  return (
    <div
      style={{ fontFamily: "'DM Sans', sans-serif" }}
      className="flex h-screen bg-[#09090e] text-white overflow-hidden"
    >
      <div className="flex-1 flex flex-col">
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
          <div className="flex items-center gap-3">
            <div
              className={`w-2 h-2 rounded-full ${
                status === "Ready" && !modelLoading ? "bg-emerald-400 animate-pulse" : "bg-amber-400"
              }`}
            />
            <span className="text-xs tracking-widest uppercase text-white/40">
              {modelLoading ? "Loading 3D asset…" : status}
            </span>
          </div>
          <div className="text-xs text-white/20 tracking-wider uppercase">
            Live AR · {activeGlass?.name}
          </div>
        </div>

        <div ref={containerRef} className="relative flex-1 bg-black overflow-hidden">
          {webcamError && (
            <div className="absolute inset-0 flex items-center justify-center z-20 bg-black/80">
              <div className="text-center space-y-3 px-8">
                <p className="text-red-400 text-sm">{webcamError}</p>
                <button
                  onClick={retryWebcam}
                  className="px-5 py-2 bg-white text-black text-xs font-bold rounded-full"
                >
                  Retry Camera
                </button>
              </div>
            </div>
          )}

          <video
            ref={videoRef}
            playsInline
            muted
            className="absolute inset-0 w-full h-full object-cover"
            style={{ transform: "scaleX(-1)" }}
          />

          <canvas
            ref={overlayCanvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none"
            style={{ transform: "scaleX(-1)" }}
          />

          {["top-4 left-4", "top-4 right-4", "bottom-4 left-4", "bottom-4 right-4"].map(
            (pos, i) => (
              <div key={i} className={`absolute ${pos} w-6 h-6 pointer-events-none`}>
                <div
                  className="absolute border-white/30"
                  style={{
                    width: 20,
                    height: 20,
                    borderTopWidth: i < 2 ? 1.5 : 0,
                    borderBottomWidth: i >= 2 ? 1.5 : 0,
                    borderLeftWidth: i % 2 === 0 ? 1.5 : 0,
                    borderRightWidth: i % 2 === 1 ? 1.5 : 0,
                  }}
                />
              </div>
            ),
          )}
        </div>

        <div className="flex items-center justify-center gap-4 px-6 py-5 border-t border-white/5">
          <button
            disabled={!activeGlass || votedGlasses.includes(activeGlass?.glass_id)}
            onClick={() => handleFeedback(false)}
            className="flex items-center gap-2 px-6 py-2.5 rounded-full border border-white/10
              text-sm font-medium text-white/60 hover:border-red-500/50 hover:text-red-400
              disabled:opacity-30 disabled:cursor-not-allowed transition-all duration-200"
          >
            <span className="text-base">👎</span> Dislike
          </button>
          <button
            disabled={!activeGlass || votedGlasses.includes(activeGlass?.glass_id)}
            onClick={() => handleFeedback(true)}
            className="flex items-center gap-2 px-6 py-2.5 rounded-full border border-white/10
              text-sm font-medium text-white/60 hover:border-emerald-500/50 hover:text-emerald-400
              disabled:opacity-30 disabled:cursor-not-allowed transition-all duration-200"
          >
            <span className="text-base">👍</span> Like
          </button>
        </div>
      </div>

      <div className="w-72 flex flex-col border-l border-white/5 bg-white/[0.02]">
        <div className="px-6 pt-6 pb-4 border-b border-white/5">
          <p className="text-[10px] tracking-widest uppercase text-white/30 mb-1">
            Face Shape
          </p>
          <h3 className="text-lg font-semibold tracking-tight">
            {data.detected_face_shape}
          </h3>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-2">
          {data.top_matches.map((glass: any, idx: number) => {
            const isActive = activeGlass?.glass_id === glass.glass_id;
            const hasVoted = votedGlasses.includes(glass.glass_id);
            return (
              <button
                key={glass.glass_id}
                onClick={() => setActiveGlass(glass)}
                className={`w-full text-left p-3.5 rounded-xl border transition-all duration-200 ${
                  isActive
                    ? "border-white/20 bg-white/5"
                    : "border-white/5 hover:border-white/10 hover:bg-white/[0.03]"
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-white/20 font-mono w-4">
                      {String(idx + 1).padStart(2, "0")}
                    </span>
                    <span className="text-sm font-medium text-white/80 truncate">
                      {glass.name}
                    </span>
                  </div>
                  {hasVoted && (
                    <span className="text-[10px] text-white/20">✓ voted</span>
                  )}
                </div>

                <div className="ml-6">
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-[2px] bg-white/5 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-white/30 rounded-full transition-all duration-500"
                        style={{ width: `${(glass.score * 100).toFixed(0)}%` }}
                      />
                    </div>
                    <span className="text-[10px] text-white/30 font-mono">
                      {(glass.score * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        <div className="px-6 py-4 border-t border-white/5">
          <p className="text-[10px] text-white/20 text-center leading-relaxed">
            Live AR · 478-point face mesh
          </p>
        </div>
      </div>
    </div>
  );
}