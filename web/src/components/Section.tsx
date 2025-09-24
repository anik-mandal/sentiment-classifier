import { motion } from 'framer-motion'
import React from 'react'

type Props = { id: string; children: React.ReactNode }

export default function Section({ id, children }: Props) {
  return (
    <section id={id} aria-label={id} className="min-h-screen w-full flex items-center">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
        className="container px-4"
      >
        {children}
      </motion.div>
    </section>
  )
}

