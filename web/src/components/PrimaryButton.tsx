import { motion } from 'framer-motion'
import React from 'react'

type Props = React.ButtonHTMLAttributes<HTMLButtonElement> & { label: string }

export default function PrimaryButton({ label, className = '', ...rest }: Props) {
  return (
    <motion.button
      whileTap={{ scale: 0.98 }}
      whileHover={{ boxShadow: '0 8px 30px rgba(34,197,94,0.35)' }}
      className={`rounded-full px-6 py-3 bg-emerald-500 text-black font-semibold ${className}`}
      {...rest}
    >
      {label}
    </motion.button>
  )
}

