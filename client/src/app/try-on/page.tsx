"use client";

import { Canvas } from "@react-three/fiber";
import { Environment } from "@react-three/drei";
import { useFaceTracking } from "../../../components/useFaceTracking"
import { Glasses } from "../../../components/Glasses";
import { Suspense } from "react";

export default function FaceFilter() {
  const { videoRef, landmarks, ready } = useFaceTracking();

  return (
    <div
      style={{
        position: "relative",
        width: "100vw",
        height: "100vh",
        background: "black",
      }}
    >
      {/* 1. The Webcam Feed (Hidden or Visible Background) */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          objectFit: "cover",
          transform: "scaleX(-1)", // Mirror the webcam
          zIndex: 0,
        }}
      />

      {/* 2. The 3D Overlay */}
      {ready && (
        <Canvas
          camera={{ position: [0, 0, 5], fov: 50 }}
          style={{ position: "absolute", top: 0, left: 0, zIndex: 1 }}
        >
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} />

          <Suspense fallback={null}>
            {/* Pass landmarks to the model */}
            <Glasses landmarks={landmarks} />
            <Environment preset="city" />
          </Suspense>
        </Canvas>
      )}

      {!ready && (
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            color: "white",
          }}
        >
          Loading AI Model...
        </div>
      )}
    </div>
  );
}
