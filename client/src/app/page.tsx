"use client";
import { useState, ChangeEvent, useEffect } from "react";

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);

  const handleImageChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedImage(e.target.files[0])
    }
  }
  useEffect(() => { 
    console.log(selectedImage)
  }, [selectedImage])
  return (
    <div>
      <input
        type="file"
        onChange={handleImageChange}
        accept="image/jpeg, image/png, image/jpg"
        id="input=file"
      />
    </div>
  );
}
