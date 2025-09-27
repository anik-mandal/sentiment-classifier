import { motion } from "framer-motion";
import clsx from "clsx";
import type { ButtonHTMLAttributes, ReactNode } from "react";

type ButtonVariant = "primary" | "ghost";

type PrimaryButtonProps = {
  children: ReactNode;
  variant?: ButtonVariant;
  loading?: boolean;
} & ButtonHTMLAttributes<HTMLButtonElement>;

const variantStyles: Record<ButtonVariant, string> = {
  primary:
    "bg-accent text-white hover:bg-accent/90 shadow-[0_0_25px_rgba(127,90,240,0.45)]",
  ghost:
    "bg-transparent text-white border border-white/20 hover:border-white/40 hover:bg-white/5",
};

const PrimaryButton = ({
  children,
  className,
  variant = "primary",
  loading = false,
  disabled,
  ...props
}: PrimaryButtonProps) => {
  return (
    <motion.button
      whileHover={{ scale: disabled ? 1 : 1.02 }}
      whileTap={{ scale: disabled ? 1 : 0.98 }}
      disabled={disabled || loading}
      className={clsx(
        "relative inline-flex items-center justify-center gap-2 rounded-full px-6 py-3 text-sm font-semibold transition-shadow duration-300 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/60",
        variantStyles[variant],
        (disabled || loading) && "cursor-not-allowed opacity-70",
        className,
      )}
      {...props}
    >
      {loading && (
        <span className="inline-flex h-4 w-4 animate-spin rounded-full border-2 border-white/70 border-t-transparent" />
      )}
      <span className="tracking-wide">{children}</span>
    </motion.button>
  );
};

export default PrimaryButton;
