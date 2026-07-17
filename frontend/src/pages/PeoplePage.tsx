import { useFaceSamples, useFaces } from '@/api/faces'
import { PageHeader } from '@/components/PageHeader'
import { Alert } from '@/components/ui/Alert'
import { Badge } from '@/components/ui/Badge'
import { Card, CardContent } from '@/components/ui/Card'
import { EmptyState } from '@/components/ui/EmptyState'
import { Input } from '@/components/ui/Input'
import { Skeleton } from '@/components/ui/Skeleton'
import { formatDate, mapRecognizeStatus } from '@/lib/utils'
import { CalendarDays, Search, User, Users } from 'lucide-react'
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
        subtitle="Kayıtlı ve anonim yüz kimliklerini görüntüleyin, arayın ve detaylarına ulaşın."
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
        <Alert variant="error" title="Kişiler alınamadı">
          {facesQuery.error.message}
        </Alert>
      ) : identities.length === 0 ? (
        <EmptyState
          icon={Users}
          title={search ? 'Sonuç bulunamadı' : 'Henüz kayıtlı kimlik yok'}
          description={
            search
              ? 'Aramanızla eşleşen bir kişi bulunamadı; farklı bir anahtar kelime deneyin.'
              : 'Yüz tanıma işlemi yaptıktan sonra kimlikler burada listelenecek.'
          }
        />
      ) : (
        <>
          <p className="text-sm text-navy-500">{identities.length} kimlik</p>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {identities.map((identity) => (
              <PeopleCard key={identity.face_id} identity={identity} />
            ))}
          </div>
        </>
      )}
    </div>
  )
}

interface PeopleCardProps {
  identity: {
    face_id: string
    status: string
    name: string | null
    created_at: string | null
  }
}

function PeopleCard({ identity }: PeopleCardProps) {
  const samplesQuery = useFaceSamples(identity.face_id)
  const firstImage = samplesQuery.data?.samples.at(0)?.image_url
  const displayName = identity.name ?? 'İsimsiz yüz'

  return (
    <Link to={`/faces/${identity.face_id}`} className="group focus-visible:outline-none">
      <Card className="h-full transition-shadow hover:shadow-md">
        <CardContent className="p-0">
          <div className="relative aspect-[4/3] overflow-hidden rounded-t-xl bg-navy-50">
            {firstImage ? (
              <img
                src={firstImage}
                alt={`${displayName} örnek fotoğrafı`}
                className="h-full w-full object-cover object-center transition-transform duration-300 group-hover:scale-105"
                loading="lazy"
              />
            ) : (
              <div className="flex h-full w-full items-center justify-center text-navy-300">
                {samplesQuery.isLoading ? (
                  <Skeleton className="h-16 w-16 rounded-full" />
                ) : (
                  <User className="h-16 w-16" aria-hidden="true" />
                )}
              </div>
            )}
            <div className="absolute right-3 top-3">
              <Badge status={identity.status}>{mapRecognizeStatus(identity.status)}</Badge>
            </div>
          </div>
          <div className="space-y-2 p-4">
            <h3 className="truncate text-sm font-semibold text-navy-900">{displayName}</h3>
            <div className="flex items-center gap-1.5 text-xs text-navy-500">
              <CalendarDays className="h-3.5 w-3.5" aria-hidden="true" />
              {identity.created_at ? formatDate(identity.created_at) : '—'}
            </div>
          </div>
        </CardContent>
      </Card>
    </Link>
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
