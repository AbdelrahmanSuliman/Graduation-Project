"use client";

import { useState, ChangeEvent, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { getRecommendations } from "../../api/api";
import { extractBiometrics } from "../../utils/geometry";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

export default function Home() {
  const router = useRouter();
  
  // App States: 'idle' | 'camera' 
  const [mode, setMode] = useState<"idle" | "camera">("idle");
  const [isProcessing, setIsProcessing] = useState(false);
  const [status, setStatus] = useState("");
  
  // Webcam and tracking refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const rafRef = useRef<number>(0);
  const faceHoldFrames = useRef<number>(0);

  // ── 1. Start Webcam & MediaPipe (Only runs when mode === 'camera') ─────
  useEffect(() => {
    let isMounted = true;

    if (mode === "camera") {
      setStatus("Initializing camera...");
      const initSystem = async () => {
        try {
          // Start Camera
          const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
          });
          streamRef.current = stream;
          
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            await new Promise<void>((resolve) => {
              if (videoRef.current) {
                videoRef.current.onloadedmetadata = () => resolve();
              }
            });
            await videoRef.current.play();
          }

          setStatus("Loading AI...");

          // Load MediaPipe
          const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
          );
          landmarkerRef.current = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
              delegate: "GPU",
            },
            runningMode: "VIDEO",
            numFaces: 1,
          });

          if (isMounted) {
            setStatus("Look at the camera");
            detectLoop();
          }

        } catch (err) {
          console.error("Initialization error:", err);
          setStatus("Camera access denied or failed.");
          setMode("idle");
        }
      };

      initSystem();
    }

    // Cleanup function: stop stream and AI if component unmounts or mode changes
    return () => {
      isMounted = false;
      cancelAnimationFrame(rafRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (landmarkerRef.current) {
        landmarkerRef.current.close();
      }
    };
  }, [mode]);

  // ── 2. Real-time Face Detection Loop ────────────────────────────────
  const detectLoop = () => {
    if (!videoRef.current || !landmarkerRef.current) return;

    const video = videoRef.current;
    
    // Only run if video is playing
    if (video.readyState >= 2) {
      const results = landmarkerRef.current.detectForVideo(video, performance.now());

      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        // Face detected! Increment hold counter
        faceHoldFrames.current += 1;
        
        // If face is held steady for ~30 frames (about 1 second), take the photo
        if (faceHoldFrames.current > 30) {
          captureAndProcess();
          return; // Stop the loop
        } else if (faceHoldFrames.current > 10) {
          setStatus("Hold still...");
        }
      } else {
        // Face lost, reset counter
        faceHoldFrames.current = 0;
        setStatus("Look at the camera");
      }
    }

    rafRef.current = requestAnimationFrame(detectLoop);
  };

  // ── 3. Capture Frame and trigger existing pipeline ──────────────────
  const captureAndProcess = () => {
    cancelAnimationFrame(rafRef.current);
    setStatus("Capturing...");
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    // Set canvas to actual video dimensions
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Draw current video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Stop the camera stream immediately so the UI freezes on the captured frame
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
    }

    // Convert canvas to a File object and pass it to your existing logic
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], "webcam_capture.jpg", { type: "image/jpeg" });
        setIsProcessing(true); // Switch to processing UI
        uploadImage(file);
      }
    }, "image/jpeg", 0.95);
  };

  // ── 4. Manual File Upload Handler ───────────────────────────────────
  const handleImageChange = async (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setIsProcessing(true);
      setMode("idle"); // Ensure camera is off
      await uploadImage(file);
    }
  };

  // ── 5. Your Existing Pipeline Logic ─────────────────────────────────
  const processImageWithMediaPipe = async (file: File) => {
    const img = new Image();
    img.src = URL.createObjectURL(file);
    await new Promise((resolve) => {
      img.onload = resolve;
    });

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
    );

    const landmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "IMAGE",
      numFaces: 1,
    });

    const results = landmarker.detect(img);
    if (!results.faceLandmarks || results.faceLandmarks.length === 0) {
      throw new Error(
        "No face detected in the image. Please try a clearer photo.",
      );
    }

    const biometrics = extractBiometrics(
      results.faceLandmarks[0],
      img.width,
      img.height,
    );

    if (!biometrics) {
      throw new Error("Could not calculate facial geometry.");
    }
    const face = results.faceLandmarks[0];
    const arAnchors = {
      leftEye: face[33],
      rightEye: face[263],
      nose: face[168]
    };
    sessionStorage.setItem("ar_anchors", JSON.stringify(arAnchors));

    return biometrics;
  };

  const uploadImage = async (file: File) => {
    try {
      setStatus("Extracting 478 biometric points & running AI...");
      
      const features = await processImageWithMediaPipe(file);
      console.log("Extracted Features:", features);

      const res = await getRecommendations({
        file: file,
        features: features,
      });

      console.log("AI Results:", res);

      if (res && res.top_matches) {
        sessionStorage.setItem("ai_recommendations", JSON.stringify(res));
        sessionStorage.setItem("user_features", JSON.stringify(features));

        const imageUrl = URL.createObjectURL(file);
        sessionStorage.setItem("user_image", imageUrl);

        router.push("/try-on");
      } else {
        throw new Error("Invalid response from server");
      }
    } catch (error: any) {
      console.error("Pipeline Error:", error);
      alert(
        error.message || "Failed to process image. Check console for details.",
      );
      setIsProcessing(false);
      setMode("idle");
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-[#0a0a0f] text-white p-6">
      <div className="max-w-2xl w-full space-y-8 text-center">
        
        {!isProcessing && mode === "idle" && (
          <div className="space-y-12 animate-in fade-in zoom-in duration-500">
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
              WELCOME TO THE FUTURE OF EYEWEAR
            </h1>
            <p className="text-lg text-white/60 max-w-xl mx-auto leading-relaxed">
              Discover frames that actually fit. Let our AI analyze 478 points on your face to find your perfect style and size in seconds.
            </p>
            
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
              <button
                onClick={() => setMode("camera")}
                className="w-full sm:w-auto px-8 py-4 rounded-full bg-white text-black font-bold shadow-xl hover:scale-105 transition-all flex items-center justify-center gap-3"
              >
                <span className="text-xl">📸</span> Take Live Photo
              </button>
              
              <label
                htmlFor="file-upload"
                className="w-full sm:w-auto cursor-pointer px-8 py-4 rounded-full bg-white/10 text-white font-bold hover:bg-white/20 border border-white/10 transition-all flex items-center justify-center gap-3"
              >
                <span className="text-xl">📁</span> Upload a Photo
              </label>
              <input
                id="file-upload"
                className="hidden"
                type="file"
                onChange={handleImageChange}
                accept="image/jpeg, image/png, image/jpg"
              />
            </div>
          </div>
        )}

        {mode === "camera" && !isProcessing && (
          <div className="space-y-6 animate-in slide-in-from-bottom-8 duration-500">
            <h2 className="text-2xl font-semibold tracking-tight">Position your face in the frame</h2>
            <p className="text-white/50 text-sm">We'll automatically capture the photo when you hold still.</p>
            
            <div className="relative mx-auto rounded-3xl overflow-hidden bg-black aspect-video border border-white/10 shadow-2xl max-w-lg">
              {/* Live Webcam Feed - Mirrored for natural feeling */}
              <video
                ref={videoRef}
                playsInline
                muted
                className="w-full h-full object-cover"
                style={{ transform: "scaleX(-1)" }}
              />
              
              {/* Status Overlay */}
              <div className="absolute inset-0 flex flex-col items-center justify-end pb-8 pointer-events-none">
                <div className="bg-black/50 backdrop-blur-md border border-white/10 px-6 py-2 rounded-full">
                  <p className="text-sm font-medium text-white/90 animate-pulse">
                    {status}
                  </p>
                </div>
              </div>
            </div>

            <button
              onClick={() => setMode("idle")}
              className="px-6 py-2 text-sm text-white/50 hover:text-white transition-colors"
            >
              Cancel Camera
            </button>
          </div>
        )}

        {/* Hidden canvas for taking the snapshot */}
        <canvas ref={canvasRef} className="hidden" />

        {isProcessing && (
          <div className="flex flex-col items-center justify-center space-y-6 py-12 animate-in fade-in duration-300">
            <div className="relative w-20 h-20">
              <div className="absolute inset-0 border-4 border-white/10 rounded-full"></div>
              <div className="absolute inset-0 border-4 border-t-white rounded-full animate-spin"></div>
            </div>
            <div className="space-y-2">
              <h3 className="text-xl font-medium tracking-tight">Analyzing Facial Geometry</h3>
              <p className="text-sm text-white/50 max-w-xs mx-auto">
                {status}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}