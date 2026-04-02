import { useState, useCallback } from 'react'
import { uploadFiles } from '../api/client'
import type { UploadResult } from '../types'

export function useFileUpload(conversationId: string) {
  const [isUploading, setIsUploading] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([])
  const [lastError, setLastError] = useState<string | null>(null)

  const upload = useCallback(async (files: File[]): Promise<UploadResult | null> => {
    if (files.length === 0) return null
    setIsUploading(true)
    setLastError(null)

    try {
      const result = await uploadFiles(files, conversationId)
      setUploadedFiles(prev => [...prev, ...result.filenames])
      if (result.errors.length > 0) {
        setLastError(result.errors.join(', '))
      }
      return result
    } catch (error) {
      const msg = (error as Error).message
      setLastError(msg)
      return null
    } finally {
      setIsUploading(false)
    }
  }, [conversationId])

  return { isUploading, uploadedFiles, lastError, upload }
}
