"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { submitFeedback } from "../../../api/api";

export default function TryOnPage() {
  const router = useRouter();
  const [data, setData] = useState<any>(null);
  const [activeGlass, setActiveGlass] = useState<any>(null);
  const [userImage, setUserImage] = useState<string>("");
  const [votedGlasses, setVotedGlasses] = useState<number[]>([]);

  useEffect(() => {
    const storedRes = sessionStorage.getItem("ai_recommendations");
    const storedImg = sessionStorage.getItem("user_image");

    if (storedRes && storedImg) {
      const parsedRes = JSON.parse(storedRes);
      setData(parsedRes);
      setActiveGlass(parsedRes.top_matches[0]);
      setUserImage(storedImg);
    } else {
      router.push("/");
    }
  }, [router]);

  const handleFeedback = async (liked: boolean) => {
    if (!data || !activeGlass) return;
    
    if (votedGlasses.includes(activeGlass.glass_id)) {
      alert("You've already provided feedback for these glasses!");
      return;
    }
    const featuresDict = JSON.parse(
      sessionStorage.getItem("user_features") || "{}",
    );
    const featuresArray = [
      featuresDict.cheek_jaw_ratio || 1.0,
      featuresDict.face_hw_ratio || 1.0,
      featuresDict.midface_ratio || 1.0,
    ];

    console.log(featuresArray)

    try {
      await submitFeedback({
        glass_id: activeGlass.glass_id,
        detected_face_shape: data.detected_face_shape_id,
        features: featuresArray,
        liked: liked,
      });
      setVotedGlasses((prev) => [...prev, activeGlass.glass_id])
      alert(
        `Feedback saved! You ${liked ? "liked" : "disliked"} these glasses.`,
      );
    } catch (error) {
      console.error(error);
      alert("Error saving feedback.");
    }
  };

  if (!data)
    return (
      <div className="flex justify-center p-20">Loading AI Results...</div>
    );

  return (
    <div className="flex h-screen bg-gray-50">
      <div className="flex-1 flex flex-col items-center justify-center relative">
        <h2 className="text-2xl font-bold mb-4 absolute top-10">
          AI Try-On Room
        </h2>

        <div className="relative w-96 h-96 bg-gray-200 border-4 border-white shadow-lg rounded overflow-hidden">
          <img
            src={userImage}
            alt="User"
            className="w-full h-full object-cover"
          />
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-white bg-black bg-opacity-50 px-4 py-2">
            [AR Overlay: {activeGlass.name}]
          </div>
        </div>

        <div className="flex gap-6 mt-8">
          <button
            disabled={votedGlasses.includes(activeGlass.glass_id)}
            onClick={() => handleFeedback(false)}
            className="px-8 py-3 bg-red-500 hover:bg-red-600 text-white rounded-full font-bold shadow transition"
          >
            👎 Dislike
          </button>
          <button
            disabled={votedGlasses.includes(activeGlass.glass_id)}
            onClick={() => handleFeedback(true)}
            className="px-8 py-3 bg-green-500 hover:bg-green-600 text-white rounded-full font-bold shadow transition"
          >
            👍 Like
          </button>
        </div>
      </div>

      <div className="w-[400px] bg-white shadow-2xl p-6 overflow-y-auto">
        <h3 className="text-xl font-bold mb-1">Your Matches</h3>
        <p className="text-sm text-gray-500 mb-6">
          Face Shape Detected:{" "}
          <span className="text-blue-600 font-bold">
            {data.detected_face_shape}
          </span>
        </p>

        <div className="space-y-4">
          {data.top_matches.map((glass: any) => (
            <div
              key={glass.glass_id}
              onClick={() => setActiveGlass(glass)}
              className={`p-4 rounded-xl border-2 cursor-pointer transition ${
                activeGlass?.glass_id === glass.glass_id
                  ? "border-blue-500 bg-blue-50 shadow-md"
                  : "border-gray-100 hover:border-gray-300"
              }`}
            >
              <div className="flex justify-between items-start">
                <h4 className="font-semibold text-gray-800">{glass.name}</h4>
                <span className="bg-green-100 text-green-800 text-xs font-bold px-2 py-1 rounded">
                  {(glass.score * 100).toFixed(1)}% Match
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
