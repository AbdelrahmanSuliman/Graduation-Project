"use client";

import { useState, ChangeEvent } from "react";
import { useRouter } from "next/navigation";
import { getRecommendations } from "../../api/api";
import { extractBiometrics } from "../../utils/geometry"; 
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const router = useRouter();

  const handleImageChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedImage(e.target.files[0]);
    }
  };

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

    return biometrics;
  };

  const uploadImage = async () => {
    if (!selectedImage) {
      alert("Please select an image first.");
      return;
    }

    setIsProcessing(true);

    try {
      const features = await processImageWithMediaPipe(selectedImage);
      console.log("Extracted Features:", features);


      const res = await getRecommendations({
        file: selectedImage,
        features: features,
      });

      console.log("AI Results:", res);

      if (res && res.recommendations) {
        sessionStorage.setItem("ai_recommendations", JSON.stringify(res));
      }

      router.push("/try-on");
    } catch (error: any) {
      console.error("Pipeline Error:", error);
      alert(
        error.message || "Failed to process image. Check console for details.",
      );
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100vh",
        gap: "20px",
      }}
    >
      <h1>Spectacular AI Try-On</h1>

      <input
        type="file"
        onChange={handleImageChange}
        accept="image/jpeg, image/png, image/jpg"
        disabled={isProcessing}
      />

      <button
        onClick={uploadImage}
        disabled={!selectedImage || isProcessing}
        style={{
          padding: "10px 20px",
          fontSize: "16px",
          cursor: isProcessing ? "not-allowed" : "pointer",
        }}
      >
        {isProcessing ? "Processing..." : "Submit & Try On"}
      </button>


    </div>
  );
}
