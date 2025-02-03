"use client";

import { motion, AnimatePresence } from "framer-motion";

interface MagicalLoadingOverlayProps {
  isLoading: boolean;
}

export const MagicalLoadingOverlay: React.FC<MagicalLoadingOverlayProps> = ({
  isLoading,
}) => {
  return (
    <AnimatePresence mode="wait">
      {isLoading && (
        <motion.div
          key="loading-overlay"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{
            type: "tween",
            duration: 1,
            ease: [0.4, 0, 0.2, 1],
          }}
          className="!transition-none"
          style={{
            position: "absolute",
            inset: 0,
            pointerEvents: "none",
            overflow: "hidden",
            willChange: "opacity",
            isolation: "isolate",
            zIndex: 50,
          }}
        >
          {/* Container with mask */}
          <div
            className="!transition-none"
            style={{
              position: "absolute",
              inset: 0,
              WebkitMask:
                "radial-gradient(circle at center, transparent 60%, black 100%)",
              mask: "radial-gradient(circle at center, transparent 60%, black 100%)",
            }}
          >
            {/* Glow effect */}
            <motion.div
              className="!transition-none"
              style={{
                position: "absolute",
                inset: "-50%",
                background: `
                  conic-gradient(
                    from 180deg at 50% 50%,
                    #2E3FE5 0deg,
                    #B829E3 75deg,
                    #FF2525 160deg,
                    #FF7A01 220deg,
                    #2E3FE5 360deg
                  )
                `,
                filter: "blur(40px) brightness(1.5)",
                opacity: 0.7,
              }}
              animate={{ rotate: 360 }}
              transition={{
                repeat: Infinity,
                duration: 10,
                ease: "linear",
              }}
            />
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
