
// Helper function to calculate 2D distance between two landmarks
const calcDistance = (p1: any, p2: any, width: number, height: number) => {
  const dx = (p1.x - p2.x) * width;
  const dy = (p1.y - p2.y) * height;
  return Math.sqrt(dx * dx + dy * dy);
};

export const extractBiometrics = (
  landmarks: any[],
  imgWidth: number,
  imgHeight: number,
) => {
  if (!landmarks || landmarks.length === 0) return null;

  // MediaPipe specific indices for key facial features
  const topHead = landmarks[10];
  const bottomChin = landmarks[152];
  const leftCheek = landmarks[234];
  const rightCheek = landmarks[454];
  const leftJaw = landmarks[132];
  const rightJaw = landmarks[361];
  const leftForehead = landmarks[103];
  const rightForehead = landmarks[332];

  // Calculate pixel distances
  const faceLength = calcDistance(topHead, bottomChin, imgWidth, imgHeight);
  const cheekWidth = calcDistance(leftCheek, rightCheek, imgWidth, imgHeight);
  const jawWidth = calcDistance(leftJaw, rightJaw, imgWidth, imgHeight);
  const foreheadWidth = calcDistance(
    leftForehead,
    rightForehead,
    imgWidth,
    imgHeight,
  );

  return {
    cheek_jaw_ratio: parseFloat((cheekWidth / jawWidth).toFixed(3)),
    face_hw_ratio: parseFloat((faceLength / cheekWidth).toFixed(3)),
    midface_ratio: parseFloat((cheekWidth / foreheadWidth).toFixed(3)),
  };
};
