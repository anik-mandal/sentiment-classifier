import { motion } from "framer-motion";
import clsx from "clsx";
import type { ReactNode } from "react";

type InputMode = "text" | "csv" | "pdf";

type InputModeCardProps = {
  mode: InputMode;
  title: string;
  description: string;
  icon: ReactNode;
  active: boolean;
  onSelect: (mode: InputMode) => void;
};

const InputModeCard = ({
  mode,
  title,
  description,
  icon,
  active,
  onSelect,
}: InputModeCardProps) => {
  return (
    <motion.button
      type="button"
      layout
      onClick={() => onSelect(mode)}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className={clsx(
        "glass-card fade-border relative flex h-full w-full flex-col items-start justify-between overflow-hidden p-6 text-left transition-all duration-300",
        active
          ? "border-white/30 bg-white/15 shadow-[0_0_35px_rgba(127,90,240,0.25)]"
          : "hover:bg-white/10",
      )}
    >
      <div className="flex items-center gap-3 text-base font-medium text-white/80">
        <span className={clsx("text-2xl", active ? "text-accent" : "text-white/60")}>{icon}</span>
        {title}
      </div>
      <p className="mt-4 text-sm leading-relaxed text-white/60">{description}</p>
      {active && (
        <motion.span
          layoutId="active-card-glow"
          className="pointer-events-none absolute inset-0 -z-10 bg-accent/15"
          style={{ filter: "blur(40px)" }}
        />
      )}
    </motion.button>
  );
};

export default InputModeCard;
