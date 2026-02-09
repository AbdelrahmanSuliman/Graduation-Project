import { useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

export const useFaceTracking = () => {
  const [landmarks, setLandmarks] = useState<any[]>([]);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [ready, setReady] = useState(false);

  // Use a ref to keep track of the landmarker instance without triggering re-renders
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);

  useEffect(() => {
    let animationFrameId: number;

    const setup = async () => {
      try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
        );

        faceLandmarkerRef.current = await FaceLandmarker.createFromOptions(
          filesetResolver,
          {
            baseOptions: {
              modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
              // delegate: "GPU" // Commented out to let MediaPipe choose best backend (prevents conflicts)
              delegate: "GPU",
            },
            outputFaceBlendshapes: true,
            runningMode: "VIDEO",
            numFaces: 1,
          },
        );

        startWebcam();
      } catch (error) {
        console.error("Error initializing MediaPipe:", error);
      }
    };

    const startWebcam = async () => {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: {
              facingMode: "user",
              width: 1280,
              height: 720,
            },
          });

          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            // Wait for data to actually load before predicting
            videoRef.current.onloadeddata = () => {
              videoRef.current!.play();
              setReady(true);
              predict();
            };
          }
        } catch (err) {
          console.error("Error accessing webcam:", err);
        }
      }
    };

    const predict = () => {
      // 1. Safety Check: Ensure video is actually playing and has data
      if (
        faceLandmarkerRef.current &&
        videoRef.current &&
        videoRef.current.readyState === 4 // HAVE_ENOUGH_DATA
      ) {
        try {
          // 2. Perform detection inside a try-catch to prevent crashes on skipped frames
          const results = faceLandmarkerRef.current.detectForVideo(
            videoRef.current,
            performance.now(),
          );

          if (results.faceLandmarks && results.faceLandmarks.length > 0) {
            setLandmarks(results.faceLandmarks[0]);
          }
        } catch (e) {
          // Occasionally MediaPipe drops a frame, simply ignore it
          console.warn("Frame drop:", e);
        }
      }

      animationFrameId = requestAnimationFrame(predict);
    };

    setup();

    return () => {
      cancelAnimationFrame(animationFrameId);
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
      }
      if (faceLandmarkerRef.current) {
        faceLandmarkerRef.current.close();
      }
    };
  }, []);

  return { videoRef, landmarks, ready };
};
