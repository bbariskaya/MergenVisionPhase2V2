import { useDeleteFaceMutation, useFaces } from '@/api/faces'
import type { IdentitySummary } from '@/api/types'
import { PageHeader } from '@/components/PageHeader'
import { Alert } from '@/components/ui/Alert'
import { Card, CardContent } from '@/components/ui/Card'
import { EmptyState } from '@/components/ui/EmptyState'
import { Input } from '@/components/ui/Input'
import { Skeleton } from '@/components/ui/Skeleton'
import { Badge } from '@/components/ui/Badge'
import { cn, formatDate, mapRecognizeStatus } from '@/lib/utils'
import { CalendarDays, Loader2, Search, Trash2, User, Users } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router'

export default function PeoplePage() {
  const [search, setSearch] = useState('')

  const facesQuery = useFaces({ search })
  const identities = facesQuery.data?.identities ?? []

  return (
    <div className="space-y-6">
      <PageHeader
        title="Kişiler"
        subtitle="Kayıtlı kimlikleri görüntüleyin, arayın ve yönetin."
        action={
          <div className="relative w-full sm:w-72">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-navy-400" aria-hidden="true" />
            <Input
              type="text"
              placeholder="Ara..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9"
              aria-label="Kişi ara"
            />
          </div>
        }
      />

      {facesQuery.isLoading ? (
        <PeopleGridSkeleton />
      ) : facesQuery.error ? (
        <Alert variant="error" title="Kimlikler alınamadı">
          {facesQuery.error.message}
        </Alert>
      ) : identities.length === 0 ? (
        <EmptyState
          icon={Users}
          title={search ? 'Sonuç bulunamadı' : 'Henüz kayıtlı kişi yok'}
          description={
            search
              ? 'Aramanızla eşleşen bir kimlik bulunamadı; farklı bir anahtar kelime deneyin.'
              : 'Yüz tanıma sonrası anonim bir yüzü kaydedin veya toplu enrollment çalıştırın.'
          }
        />
      ) : (
        <>
          <p className="text-sm text-navy-500">{identities.length} kişi</p>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {identities.map((identity) => (
              <IdentityCard key={identity.face_id} identity={identity} />
            ))}
          </div>
        </>
      )}
    </div>
  )
}

function IdentityCard({ identity }: { identity: IdentitySummary }) {
  const deleteFace = useDeleteFaceMutation()

  function handleDelete(e: React.MouseEvent) {
    e.preventDefault()
    e.stopPropagation()
    if (!window.confirm(`${identity.name ?? 'İsimsiz yüz'} kimliğini silmek istediğinize emin misiniz?`)) return
    deleteFace.mutate(identity.face_id)
  }

  return (
    <Card className={cn('h-full transition-shadow hover:shadow-md')}>
      <CardContent className="p-0">
        <div className="relative aspect-[4/3] overflow-hidden rounded-t-xl bg-navy-50">
          <div className="flex h-full w-full items-center justify-center text-navy-300">
            <User className="h-16 w-16" aria-hidden="true" />
          </div>
          <div className="absolute right-3 top-3 flex gap-2">
            <button
              type="button"
              onClick={handleDelete}
              disabled={deleteFace.isPending}
              className="rounded-full bg-white/90 p-1.5 text-navy-700 shadow-sm hover:bg-danger hover:text-white disabled:opacity-50"
              aria-label="Sil"
            >
              {deleteFace.isPending ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Trash2 className="h-3.5 w-3.5" />
              )}
            </button>
          </div>
        </div>
        <Link
          to={`/faces/${identity.face_id}`}
          className="group block p-4 focus-visible:outline-none"
          aria-label={identity.name ?? 'İsimsiz yüz'}
        >
          <div className="flex items-center justify-between gap-2">
            <h3 className="truncate text-sm font-semibold text-navy-900 group-hover:text-primary">
              {identity.name ?? 'İsimsiz yüz'}
            </h3>
            <Badge status={identity.status}>{mapRecognizeStatus(identity.status)}</Badge>
          </div>
          <div className="mt-2 flex items-center gap-1.5 text-xs text-navy-500">
            <CalendarDays className="h-3.5 w-3.5" aria-hidden="true" />
            {identity.created_at ? formatDate(identity.created_at) : '—'}
          </div>
        </Link>
      </CardContent>
    </Card>
  )
}

function PeopleGridSkeleton() {
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {Array.from({ length: 8 }).map((_, i) => (
        <Card key={i}>
          <CardContent className="p-0">
            <Skeleton className="aspect-[4/3] w-full rounded-t-xl" />
            <div className="space-y-2 p-4">
              <Skeleton className="h-4 w-2/3" />
              <Skeleton className="h-3 w-1/2" />
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
