import api from "lib/axios";

type Recommendation = {
  file: File ;
  features: {
    cheek_jaw_ratio: number;
    face_hw_ratio: number;
    midface_ratio: number;
  };
};

export const getRecommendations = async ({
  file,
  features,
}: Recommendation) => {
  const formData = new FormData();

  formData.append("file", file);
  formData.append("features", JSON.stringify(features));

  console.log(formData.get("file"));
  console.log(formData.get("features"));

  const res = await api.post("/recommend", formData);

  return res.data;
};