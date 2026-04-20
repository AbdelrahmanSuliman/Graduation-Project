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
  LEFT_PUPIL: 468,
  RIGHT_PUPIL: 473,
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
  
  // Model Orientation Toggle
  const [isFlipped, setIsFlipped] = useState(false);
  const isFlippedRef = useRef(false);

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

  // Toggle model flip dynamically
  const handleFlipModel = () => {
    isFlippedRef.current = !isFlippedRef.current;
    setIsFlipped(isFlippedRef.current);
  };

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
    if (!data) return;

    const canvas = overlayCanvasRef.current;
    if (!canvas) return;

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

      // Enhance lighting so the clear glass material catches reflections
      scene.add(new THREE.AmbientLight(0xffffff, 2.0));
      const dirLight = new THREE.DirectionalLight(0xffffff, 3.5);
      dirLight.position.set(2, 5, 10);
      scene.add(dirLight);
      const fillLight = new THREE.DirectionalLight(0xeef4ff, 2.5);
      fillLight.position.set(-5, -2, 5);
      scene.add(fillLight);

      rendererRef.current = renderer;
      sceneRef.current = scene;
      cameraRef.current = camera;
      setSceneReady(true);
      
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

            const mapC = (raw: any) => ({
              x: (raw.x * renderedW - xOffset) / wC,
              y: (raw.y * renderedH - yOffset) / hC,
              z: raw.z,
            });

            const leftTempleRaw = marks[LM.LEFT_TEMPLE];
            const rightTempleRaw = marks[LM.RIGHT_TEMPLE];
            const noseRaw = marks[LM.NOSE_BRIDGE];
            const leftPupilRaw = marks[LM.LEFT_PUPIL];
            const rightPupilRaw = marks[LM.RIGHT_PUPIL];

            const leftTempleW = toWorld(mapC(leftTempleRaw).x, mapC(leftTempleRaw).y, camera);
            const rightTempleW = toWorld(mapC(rightTempleRaw).x, mapC(rightTempleRaw).y, camera);
            const leftPupilW = toWorld(mapC(leftPupilRaw).x, mapC(leftPupilRaw).y, camera);
            const rightPupilW = toWorld(mapC(rightPupilRaw).x, mapC(rightPupilRaw).y, camera);

            const eyeMidX = (leftPupilW.x + rightPupilW.x) / 2;
            const eyeMidY = (leftPupilW.y + rightPupilW.y) / 2;
            const faceWidth = leftTempleW.distanceTo(rightTempleW);

            // Shift the glasses down slightly to rest on the nose bridge.
            // Using a percentage of faceWidth keeps the offset accurate at any distance.
            const verticalOffset = faceWidth * 0.065; 
            const finalY = eyeMidY - verticalOffset;

            const dx = leftPupilW.x - rightPupilW.x;
            const dy = leftPupilW.y - rightPupilW.y;
            const roll = Math.atan2(dy, dx);
            
            const pitch = (noseRaw.z - (leftPupilRaw.z + rightPupilRaw.z) * 0.5) * 2.5;
            const faceCenterRawX = (leftTempleRaw.x + rightTempleRaw.x) / 2;
            const yaw = -(noseRaw.x - faceCenterRawX) * Math.PI * 1.5;

            // Apply orientation offset if user clicked the Flip button
            const yawOffset = isFlippedRef.current ? Math.PI : 0;

            glasses.position.set(eyeMidX, finalY, 0); // Replaced eyeMidY with finalY
            glasses.scale.setScalar(faceWidth * 1.05); 
            glasses.rotation.set(pitch, yaw + yawOffset, roll);
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

      return () => {
        cancelAnimationFrame(rafRef.current);
        window.removeEventListener("resize", handleResize);
        streamRef.current?.getTracks().forEach((t) => t.stop());
        renderer.dispose();
        landmarkerRef.current?.close();
      };
    }, 50);

    return () => clearTimeout(timeoutId);
  }, [data]);

  // ── 3. Load / swap glasses model ───────────────────────────────────────────
  useEffect(() => {
    if (!sceneReady || !sceneRef.current || !activeGlass) return;

    if (glassesGroupRef.current) {
      sceneRef.current.remove(glassesGroupRef.current);
      glassesGroupRef.current = null;
    }

    // Reset flip toggle when switching to a new glass model
    isFlippedRef.current = false;
    setIsFlipped(false);
    setModelLoading(true);

    const fileName =
      activeGlass.file_name ||
      `glass_${String(activeGlass.glass_id + 1).padStart(2, "0")}`;
    const modelPath = `/models/${fileName}.obj`;

    loaderRef.current.load(
      modelPath,
      (obj) => {
        const mainGroup = new THREE.Group();

        // Material 1: Standard Frame
        const frameMat = new THREE.MeshPhysicalMaterial({
          color: 0x18181b, // Dark charcoal
          metalness: 0.6,
          roughness: 0.3,
          clearcoat: 1.0,
          transparent: true,
          opacity: 0.85, // Just slightly transparent as a fallback if everything is one mesh
        });

        // Material 2: Pure Clear Glass for Lenses
        const lensMat = new THREE.MeshPhysicalMaterial({
          color: 0xffffff,
          transmission: 1.0,     // 100% transparent refraction
          opacity: 1.0,          // Opacity must be 1.0 for physical transmission to look right
          metalness: 0.1,
          roughness: 0.05,       // Smooth, clear glass
          ior: 1.5,              // Refractive index of glass
          thickness: 0.2,        // Volumetric thickness
          transparent: true,
        });

        const box = new THREE.Box3().setFromObject(obj);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());

        obj.traverse((child: any) => {
          if (child.isMesh) {
            const name = child.name.toLowerCase();
            // Automatically detect if this specific mesh is a lens
            if (name.includes("lens") || name.includes("glass") || name.includes("lenses")) {
              child.material = lensMat;
            } else {
              child.material = frameMat;
            }
            
            child.castShadow = true;
            child.receiveShadow = true;
            
            // 1. Shift vertices directly to perfect center
            child.geometry.translate(-center.x, -center.y, -center.z);
            // 2. Fix the upside-down issue by baking the Z rotation into the geometry
            child.geometry.rotateZ(Math.PI);
          }
        });

        if (size.x > 0) {
          obj.scale.setScalar(1 / size.x);
        }

        mainGroup.add(obj);
        mainGroup.visible = false;
        
        sceneRef.current!.add(mainGroup);
        glassesGroupRef.current = mainGroup;
        setModelLoading(false);
      },
      undefined,
      (err) => {
        console.error("OBJ load error:", err);
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
      setWebcamError("Camera access denied.");
    }
  }, []);

  // ── Feedback handler ──────────────────────────────────────────────────────
  const handleFeedback = async (liked: boolean) => {
    if (!data || !activeGlass) return;
    if (votedGlasses.includes(activeGlass.glass_id)) return;
    
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
    } catch (err) {
      console.error(err);
    }
  };

  if (!data) {
    return (
      <div className="flex items-center justify-center h-screen bg-[#0a0a0f] text-white">
        <div className="text-center space-y-3">
          <div className="w-10 h-10 border-2 border-white/20 border-t-white rounded-full animate-spin mx-auto" />
        </div>
      </div>
    );
  }

  return (
    <div
      style={{ fontFamily: "'DM Sans', sans-serif" }}
      className="flex h-screen bg-[#09090e] text-white overflow-hidden"
    >
      <div className="flex-1 flex flex-col relative">
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
              <button
                onClick={retryWebcam}
                className="px-5 py-2 bg-white text-black text-xs font-bold rounded-full"
              >
                Retry Camera
              </button>
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
          
          {/* Flip Model Orientation Button */}
          {status === "Ready" && !modelLoading && (
            <button
              onClick={handleFlipModel}
              className="absolute bottom-6 right-6 px-4 py-2 bg-white/10 hover:bg-white/20 
                rounded-full text-xs font-medium text-white backdrop-blur-md border border-white/20 
                transition-all z-10 flex items-center gap-2"
            >
              <span>🔄</span> {isFlipped ? "Unflip Glasses" : "Flip Glasses 180°"}
            </button>
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
            👎 Dislike
          </button>
          <button
            disabled={!activeGlass || votedGlasses.includes(activeGlass?.glass_id)}
            onClick={() => handleFeedback(true)}
            className="flex items-center gap-2 px-6 py-2.5 rounded-full border border-white/10
              text-sm font-medium text-white/60 hover:border-emerald-500/50 hover:text-emerald-400
              disabled:opacity-30 disabled:cursor-not-allowed transition-all duration-200"
          >
            👍 Like
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
            return (
              <button
                key={glass.glass_id}
                onClick={() => setActiveGlass(glass)}
                className={`group w-full flex items-center justify-between px-4 py-3 rounded-lg border transition-all duration-200 ${
                  isActive
                    ? "border-emerald-500/30 bg-emerald-500/10 shadow-sm"
                    : "border-white/5 hover:border-white/15 hover:bg-white/5"
                }`}
              >
                <div className="flex items-center gap-3 overflow-hidden">
                  <span className={`text-xs font-mono shrink-0 ${isActive ? "text-emerald-400" : "text-white/30 group-hover:text-white/50"}`}>
                    {String(idx + 1).padStart(2, "0")}
                  </span>
                  <span className={`text-sm font-medium truncate ${isActive ? "text-white" : "text-white/60 group-hover:text-white/90"}`}>
                    {glass.name}
                  </span>
                </div>
                {isActive && (
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0" />
                )}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}