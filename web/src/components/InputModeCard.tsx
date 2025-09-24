import { motion } from 'framer-motion'
import React from 'react'

type Props = {
  title: string
  active: boolean
  onClick: () => void
  children?: React.ReactNode
}

export default function InputModeCard({ title, active, onClick, children }: Props) {
  return (
    <motion.div
      onClick={onClick}
      className={`glass p-4 cursor-pointer ${active ? 'ring-2 ring-emerald-400' : 'opacity-80 hover:opacity-100'}`}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.99 }}
    >
      <div className="text-lg font-semibold mb-2">{title}</div>
      {children}
    </motion.div>
  )
}

