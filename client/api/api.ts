
const API_BASE = "http://localhost:8000";

export const getRecommendations = async ({
  file,
  features,
}: {
  file: File;
  features: any;
}) => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("features", JSON.stringify(features));

  const response = await fetch(`${API_BASE}/recommend`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Failed to get recommendations from the server.");
  }
  return response.json();
};

export const submitFeedback = async (feedbackData: {
  glass_id: number;
  detected_face_shape: number;
  features: number[]; 
  liked: boolean;
}) => {
  const response = await fetch(`${API_BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(feedbackData),
  });

  if (!response.ok) throw new Error("Failed to submit feedback.");
  return response.json();
};
