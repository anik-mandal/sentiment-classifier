import { motion } from "framer-motion";
import type { PropsWithChildren } from "react";
import clsx from "clsx";

type SectionProps = PropsWithChildren<{
  direction: number;
  className?: string;
  id: number;
}>;

const slideVariants = {
  initial: (direction: number) => ({
    x: direction > 0 ? "100%" : "-100%",
    opacity: 0,
    scale: 0.98,
  }),
  animate: {
    x: "0%",
    opacity: 1,
    scale: 1,
    transition: {
      duration: 0.6,
      ease: [0.22, 1, 0.36, 1],
    },
  },
  exit: (direction: number) => ({
    x: direction > 0 ? "-10%" : "10%",
    opacity: 0,
    scale: 0.98,
    transition: {
      duration: 0.4,
      ease: [0.4, 0, 0.2, 1],
    },
  }),
};

const Section = ({ children, direction, className, id }: SectionProps) => {
  return (
    <motion.section
      key={id}
      className={clsx(
        "absolute inset-0 flex h-full w-full flex-col items-center justify-center overflow-hidden px-6 md:px-12",
        className,
      )}
      custom={direction}
      variants={slideVariants}
      initial="initial"
      animate="animate"
      exit="exit"
    >
      {children}
    </motion.section>
  );
};

export default Section;
