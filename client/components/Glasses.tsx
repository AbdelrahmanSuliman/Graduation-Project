import { useGLTF } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { useRef } from "react";
import * as THREE from "three";

// --- KEY LANDMARKS (MediaPipe FaceMesh) ---
// 168: Upper Nose Bridge (Where glasses sit)
// 454: Left Face Edge (Temple)
// 234: Right Face Edge (Temple)
// 10:  Top of Head (For orientation)
// 152: Bottom of Chin (For orientation)

export function Glasses({ landmarks }: { landmarks: any[] }) {
  // 1. Load Model
  const { scene } = useGLTF("/models/glasses_1.glb");
  const groupRef = useRef<THREE.Group>(null);
  const { viewport } = useThree(); // Access 3D canvas dimensions

  useFrame(() => {
    if (!landmarks || landmarks.length === 0 || !groupRef.current) return;

    // --- 1. GET CRITICAL POINTS ---
    const noseBridge = landmarks[168]; // The Anchor
    const leftTemple = landmarks[234]; // Right side on screen (Mirror)
    const rightTemple = landmarks[454]; // Left side on screen (Mirror)

    // --- 2. POSITION (Anchor to Nose Bridge) ---
    // Map Normalized (0-1) to Viewport (-width/2 to +width/2)
    // Note: We flip X because of the mirror effect
    const x = (noseBridge.x - 0.5) * viewport.width * -1;
    const y = -(noseBridge.y - 0.5) * viewport.height;

    // Z-DEPTH: This is the "secret sauce".
    // MediaPipe Z is normalized by face width. We multiply strictly to pull it
    // forward/back. A factor of 5-10 usually looks correct in R3F.
    const z = -noseBridge.z * 5;

    groupRef.current.position.set(x, y, z);

    // --- 3. SCALE (Fit Face Width) ---
    // Calculate distance between temples to ensure arms reach ears
    const faceWidth = Math.sqrt(
      Math.pow(rightTemple.x - leftTemple.x, 2) +
        Math.pow(rightTemple.y - leftTemple.y, 2),
    );

    // SCALE MULTIPLIER:
    // If your model is in meters, this might be ~5-6.
    // If it's huge, this might be ~0.01. Tweak this ONE number until it fits.
    const scale = viewport.width * faceWidth * 0.4;
    groupRef.current.scale.set(scale, scale, scale);

    // --- 4. ROTATION (Quaternion is smoother than Euler) ---
    // We construct a "Look Matrix" based on the face orientation

    // A. Calculate the vector pointing from Left Temple to Right Temple (X-Axis)
    const rightVec = new THREE.Vector3(
      (rightTemple.x - leftTemple.x) * -1, // Flip X for mirror
      -(rightTemple.y - leftTemple.y), // Flip Y for 3D
      (rightTemple.z - leftTemple.z) * 5, // Amplify Z for noticeable turning
    ).normalize();

    // B. Calculate "Up" vector (Nose Bridge to Top of Head)
    const topHead = landmarks[10];
    const upVec = new THREE.Vector3(
      (topHead.x - noseBridge.x) * -1,
      -(topHead.y - noseBridge.y),
      (topHead.z - noseBridge.z) * 5,
    ).normalize();

    // C. Calculate "Forward" vector (Z-Axis) via Cross Product
    const forwardVec = new THREE.Vector3()
      .crossVectors(rightVec, upVec)
      .normalize();

    // D. Create Rotation Matrix
    const rotationMatrix = new THREE.Matrix4();
    rotationMatrix.makeBasis(rightVec, upVec, forwardVec);

    // Apply rotation
    groupRef.current.quaternion.setFromRotationMatrix(rotationMatrix);
  });

  return (
    // 1. OUTER GROUP: Tracks the face (Position/Rotation/Scale)
    <group ref={groupRef}>
      {/* 2. INNER MODEL: Fixes the asset direction */}
      {/* rotation={[0, Math.PI, 0]} rotates it 180 degrees around Y axis */}
      <primitive object={scene} rotation={[0, Math.PI, 0]} />
    </group>
  );
}
