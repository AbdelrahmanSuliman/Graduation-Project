// client/utils/geometry.ts

// Standard MediaPipe Face Mesh Indices for our specific anchor points
export const LANDMARKS = {
  CHEEK_LEFT: 234, // Maximum width of the left cheekbone
  CHEEK_RIGHT: 454, // Maximum width of the right cheekbone
  JAW_LEFT: 58, // Left jaw angle
  JAW_RIGHT: 288, // Right jaw angle
  FOREHEAD: 10, // Top edge of the face (hairline)
  CHIN: 152, // Absolute bottom of the chin
  NOSE_TIP: 1, // Tip of the nose
};

/**
 * Calculates the 3D Euclidean distance between two MediaPipe landmarks in pixel space.
 * Un-normalizes coordinates using video dimensions to ensure X and Y scales match.
 */
const getDistance = (p1: any, p2: any, width: number, height: number) => {
  const x1 = p1.x * width;
  const y1 = p1.y * height;
  const z1 = p1.z * width; // Z is typically scaled proportionally to width by MediaPipe

  const x2 = p2.x * width;
  const y2 = p2.y * height;
  const z2 = p2.z * width;

  return Math.sqrt(
    Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2) + Math.pow(z2 - z1, 2),
  );
};

/**
 * Extracts the 3 core biometric ratios required by the HybridNeuMF AI Model.
 */
export const extractBiometrics = (
  landmarks: any[],
  videoWidth = 1280,
  videoHeight = 720,
) => {
  if (!landmarks || landmarks.length === 0) return null;

  // 1. Calculate Raw Distances
  const cheekWidth = getDistance(
    landmarks[LANDMARKS.CHEEK_LEFT],
    landmarks[LANDMARKS.CHEEK_RIGHT],
    videoWidth,
    videoHeight,
  );
  const jawWidth = getDistance(
    landmarks[LANDMARKS.JAW_LEFT],
    landmarks[LANDMARKS.JAW_RIGHT],
    videoWidth,
    videoHeight,
  );
  const faceHeight = getDistance(
    landmarks[LANDMARKS.FOREHEAD],
    landmarks[LANDMARKS.CHIN],
    videoWidth,
    videoHeight,
  );
  const midfaceHeight = getDistance(
    landmarks[LANDMARKS.FOREHEAD],
    landmarks[LANDMARKS.NOSE_TIP],
    videoWidth,
    videoHeight,
  );

  // Safety check to prevent division by zero
  if (jawWidth === 0 || cheekWidth === 0 || faceHeight === 0) return null;

  // 2. Compute Ratios (Scale Invariant)
  const cheek_jaw_ratio = cheekWidth / jawWidth;
  const face_hw_ratio = faceHeight / cheekWidth;
  const midface_ratio = midfaceHeight / faceHeight;

  // 3. Return cleaned up numbers (rounded to 3 decimal places for clean JSON payload)
  return {
    cheek_jaw_ratio: Number(cheek_jaw_ratio.toFixed(3)),
    face_hw_ratio: Number(face_hw_ratio.toFixed(3)),
    midface_ratio: Number(midface_ratio.toFixed(3)),
  };
};
