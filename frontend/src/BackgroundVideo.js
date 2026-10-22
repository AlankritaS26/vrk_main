import React, { useEffect, useState } from 'react';

const IDLE_PHOTOS = [
  '/media/rnsit-cse-block.png',
  '/media/rnsit-amphitheatre.png',
  '/media/rnsit-mechanical-block.png',
  '/media/rnsit-auditorium.png',
  '/media/rnsit-library.png',
  '/media/rnsit-campus.png',
];

/**
 * Rotating photo background used only on the idle attract screen.
 * The local poster remains visible while each remote photo loads, so the
 * idle screen never goes blank if a remote image is unavailable.
 */
export default function IdlePhotoBackground() {
  const [photoIndex, setPhotoIndex] = useState(0);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    setLoaded(false);
    const image = new Image();
    image.onload = () => setLoaded(true);
    image.onerror = () => setLoaded(false);
    image.src = IDLE_PHOTOS[photoIndex];
  }, [photoIndex]);

  useEffect(() => {
    const rotate = setInterval(() => {
      setPhotoIndex(index => (index + 1) % IDLE_PHOTOS.length);
    }, 3000);
    return () => clearInterval(rotate);
  }, []);

  return (
    <div
      aria-hidden="true"
      style={{
        position: 'absolute',
        inset: 0,
        overflow: 'hidden',
        zIndex: 0,
        pointerEvents: 'none',
      }}
    >
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundImage: `url("${IDLE_PHOTOS[photoIndex]}")`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          opacity: loaded ? 1 : 0,
          transition: 'opacity 1s ease',
          filter: 'contrast(105%) saturate(108%) brightness(0.82)',
        }}
      />

      {!loaded && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            backgroundImage: "url('/media/college-poster.svg')",
            backgroundSize: 'cover',
            backgroundPosition: 'center',
          }}
        />
      )}

      <div
        style={{
          position: 'absolute',
          inset: 0,
          background:
            'linear-gradient(180deg, rgba(4,13,38,0.34) 0%, rgba(4,13,38,0.18) 50%, rgba(4,13,38,0.48) 100%)',
        }}
      />
    </div>
  );
}
