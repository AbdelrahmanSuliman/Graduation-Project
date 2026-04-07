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

  const handleImageChange = async (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(file);
      await uploadImage(file);
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
    const face = results.faceLandmarks[0];
    const arAnchors = {
      leftEye: face[33],
      rightEye: face[263],
      nose: face[168]
    };
    sessionStorage.setItem("ar_anchors", JSON.stringify(arAnchors));

    return biometrics;
  };

  const uploadImage = async (fileToProcess?: File) => {
    // 1. Always use the passed file to avoid React state delays
    const file = fileToProcess || selectedImage;
    if (!file) {
      alert("Please select an image first.");
      return;
    }

    setIsProcessing(true);

    try {
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
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex flex-col items-center p-12 space-y-8">
      <h1 className="text-3xl font-bold text-center max-w-2xl">
        WELCOME TO THE FUTURE OF EYEWEAR RECOMMENDATIONS
      </h1>

      <div className="flex flex-col items-center space-y-4">
        {/* We use a styled label linked to the hidden input for a better UI */}
        <label
          htmlFor="file"
          className={`cursor-pointer px-8 py-4 rounded-full bg-blue-600 text-white font-bold shadow-lg hover:bg-blue-700 transition ${isProcessing ? "opacity-50 pointer-events-none" : ""}`}
        >
          {isProcessing ? "Processing Face..." : "Upload Photo"}
        </label>

        <input
          id="file"
          className="hidden"
          type="file"
          onChange={handleImageChange}
          accept="image/jpeg, image/png, image/jpg"
          disabled={isProcessing}
        />

        {isProcessing && (
          <div className="flex flex-col items-center space-y-2 mt-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <p className="text-sm text-gray-500 font-medium">
              Extracting 478 biometric points & running AI...
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
