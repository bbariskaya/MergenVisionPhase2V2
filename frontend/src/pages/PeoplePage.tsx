import {
  useCreatePeopleBatchMutation,
  useCreatePersonMutation,
  useDeletePersonMutation,
  usePeople,
  useUpdatePersonMutation,
} from '@/api/people'
import { PageHeader } from '@/components/PageHeader'
import { Alert } from '@/components/ui/Alert'
import { Button } from '@/components/ui/Button'
import { Card, CardContent } from '@/components/ui/Card'
import { EmptyState } from '@/components/ui/EmptyState'
import { Input } from '@/components/ui/Input'
import { Modal } from '@/components/ui/Modal'
import { Skeleton } from '@/components/ui/Skeleton'
import { cn, formatDate } from '@/lib/utils'
import {
  CalendarDays,
  Edit2,
  Loader2,
  Plus,
  Search,
  Trash2,
  User,
  Users,
} from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router'

export default function PeoplePage() {
  const [search, setSearch] = useState('')
  const [isCreateOpen, setIsCreateOpen] = useState(false)
  const [isBatchOpen, setIsBatchOpen] = useState(false)
  const [editing, setEditing] = useState<{ person_id: string; display_name: string; metadata_text: string } | null>(null)

  const peopleQuery = usePeople(search)
  const people = peopleQuery.data?.people ?? []
  const createPerson = useCreatePersonMutation()
  const createPeopleBatch = useCreatePeopleBatchMutation()
  const updatePerson = useUpdatePersonMutation()

  return (
    <div className="space-y-6">
      <PageHeader
        title="Kişiler"
        subtitle="Kayıtlı kişileri görüntüleyin, arayın ve yönetin."
        action={
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
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
            <Button onClick={() => setIsBatchOpen(true)} variant="secondary">
              <Users className="mr-2 h-4 w-4" />
              Toplu Ekle
            </Button>
            <Button onClick={() => setIsCreateOpen(true)}>
              <Plus className="mr-2 h-4 w-4" />
              Yeni Kişi
            </Button>
          </div>
        }
      />

      {peopleQuery.isLoading ? (
        <PeopleGridSkeleton />
      ) : peopleQuery.error ? (
        <Alert variant="error" title="Kişiler alınamadı">
          {peopleQuery.error.message}
        </Alert>
      ) : people.length === 0 ? (
        <EmptyState
          icon={Users}
          title={search ? 'Sonuç bulunamadı' : 'Henüz kayıtlı kişi yok'}
          description={
            search
              ? 'Aramanızla eşleşen bir kişi bulunamadı; farklı bir anahtar kelime deneyin.'
              : 'Yeni kişi ekleyin veya yüz tanıma sonrası anonim bir yüzü kaydedin.'
          }
        />
      ) : (
        <>
          <p className="text-sm text-navy-500">{people.length} kişi</p>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {people.map((person) => (
              <PeopleCard
                key={person.person_id}
                person={person}
                onEdit={() =>
                  setEditing({
                    person_id: person.person_id,
                    display_name: person.display_name,
                    metadata_text: '',
                  })
                }
              />
            ))}
          </div>
        </>
      )}

      <PersonModal
        open={isCreateOpen}
        onClose={() => setIsCreateOpen(false)}
        title="Yeni Kişi Ekle"
        submitLabel="Oluştur"
        isPending={createPerson.isPending}
        onSubmit={(body) => createPerson.mutateAsync(body)}
      />

      {editing && (
        <PersonModal
          open
          onClose={() => setEditing(null)}
          title="Kişiyi Düzenle"
          submitLabel="Kaydet"
          initialDisplayName={editing.display_name}
          isPending={updatePerson.isPending}
          onSubmit={(body) => updatePerson.mutateAsync({ personId: editing.person_id, body })}
        />
      )}

      <BatchPeopleModal
        open={isBatchOpen}
        onClose={() => setIsBatchOpen(false)}
        isPending={createPeopleBatch.isPending}
        onSubmit={(body) => createPeopleBatch.mutateAsync(body)}
      />
    </div>
  )
}

function PeopleCard({
  person,
  onEdit,
}: {
  person: {
    person_id: string
    display_name: string
    is_active: boolean
    created_at: string
  }
  onEdit: () => void
}) {
  const deletePerson = useDeletePersonMutation()

  function handleDelete(e: React.MouseEvent) {
    e.preventDefault()
    e.stopPropagation()
    if (!window.confirm(`${person.display_name} kişisini silmek istediğinize emin misiniz?`)) return
    deletePerson.mutate(person.person_id)
  }

  function handleEdit(e: React.MouseEvent) {
    e.preventDefault()
    e.stopPropagation()
    onEdit()
  }

  return (
    <Card className={cn('h-full transition-shadow hover:shadow-md', !person.is_active && 'opacity-60')}>
      <CardContent className="p-0">
        <div className="relative aspect-[4/3] overflow-hidden rounded-t-xl bg-navy-50">
          <div className="flex h-full w-full items-center justify-center text-navy-300">
            <User className="h-16 w-16" aria-hidden="true" />
          </div>
          <div className="absolute right-3 top-3 flex gap-2">
            <button
              type="button"
              onClick={handleEdit}
              className="rounded-full bg-white/90 p-1.5 text-navy-700 shadow-sm hover:bg-primary hover:text-white"
              aria-label="Düzenle"
            >
              <Edit2 className="h-3.5 w-3.5" />
            </button>
            <button
              type="button"
              onClick={handleDelete}
              disabled={deletePerson.isPending}
              className="rounded-full bg-white/90 p-1.5 text-navy-700 shadow-sm hover:bg-danger hover:text-white disabled:opacity-50"
              aria-label="Sil"
            >
              {deletePerson.isPending ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Trash2 className="h-3.5 w-3.5" />
              )}
            </button>
          </div>
        </div>
        <Link to={`/people/${person.person_id}`} className="group block p-4 focus-visible:outline-none">
          <h3 className="truncate text-sm font-semibold text-navy-900 group-hover:text-primary">
            {person.display_name}
          </h3>
          <div className="mt-2 flex items-center gap-1.5 text-xs text-navy-500">
            <CalendarDays className="h-3.5 w-3.5" aria-hidden="true" />
            {person.created_at ? formatDate(person.created_at) : '—'}
          </div>
        </Link>
      </CardContent>
    </Card>
  )
}

interface PersonModalProps {
  open: boolean
  onClose: () => void
  title: string
  submitLabel: string
  initialDisplayName?: string
  isPending: boolean
  onSubmit: (body: { display_name: string; metadata?: Record<string, unknown> }) => Promise<unknown>
}

function PersonModal({ open, onClose, title, submitLabel, initialDisplayName = '', isPending, onSubmit }: PersonModalProps) {
  const [displayName, setDisplayName] = useState(initialDisplayName)
  const [metadataText, setMetadataText] = useState('')
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    if (!displayName.trim()) return

    let metadata: Record<string, unknown> | undefined
    if (metadataText.trim()) {
      try {
        metadata = JSON.parse(metadataText.trim()) as Record<string, unknown>
      } catch {
        setError('Metadata geçerli bir JSON olmalıdır.')
        return
      }
    }

    const body = {
      display_name: displayName.trim(),
      ...(metadata !== undefined && { metadata }),
    }

    try {
      await onSubmit(body)
      onClose()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'İşlem başarısız.')
    }
  }

  return (
    <Modal open={open} onClose={onClose} title={title}>
      <form onSubmit={handleSubmit} className="space-y-4">
        <Input
          label="Ad Soyad"
          value={displayName}
          onChange={(e) => setDisplayName(e.target.value)}
          placeholder="Örn: Ahmet Yılmaz"
          required
          autoComplete="name"
        />
        <div>
          <label htmlFor="person-metadata" className="label">Metadata (isteğe bağlı JSON)</label>
          <textarea
            id="person-metadata"
            value={metadataText}
            onChange={(e) => setMetadataText(e.target.value)}
            placeholder='{"department": "IT"}'
            className="input min-h-[100px] font-mono text-sm"
          />
        </div>
        {error && (
          <Alert variant="error" title="Hata">
            {error}
          </Alert>
        )}
        <div className="flex justify-end gap-3">
          <Button type="button" variant="secondary" onClick={onClose}>
            İptal
          </Button>
          <Button type="submit" isLoading={isPending} disabled={!displayName.trim()}>
            {submitLabel}
          </Button>
        </div>
      </form>
    </Modal>
  )
}

interface BatchPeopleModalProps {
  open: boolean
  onClose: () => void
  isPending: boolean
  onSubmit: (body: { people: { display_name: string }[] }) => Promise<unknown>
}

function BatchPeopleModal({ open, onClose, isPending, onSubmit }: BatchPeopleModalProps) {
  const [namesText, setNamesText] = useState('')
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)

    const names = namesText
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)

    if (names.length === 0) {
      setError('En az bir isim girin.')
      return
    }
    if (names.length > 1000) {
      setError('En fazla 1000 kişi ekleyebilirsiniz.')
      return
    }

    try {
      await onSubmit({ people: names.map((name) => ({ display_name: name })) })
      setNamesText('')
      onClose()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'İşlem başarısız.')
    }
  }

  return (
    <Modal open={open} onClose={onClose} title="Toplu Kişi Ekle">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="batch-names" className="label">
            Kişi Adları <span className="text-navy-400">(her satır bir kişi)</span>
          </label>
          <textarea
            id="batch-names"
            value={namesText}
            onChange={(e) => setNamesText(e.target.value)}
            placeholder="Ahmet Yılmaz&#10;Mehmet Demir&#10;Ayşe Kaya"
            className="input min-h-[180px] font-mono text-sm"
          />
        </div>
        {error && (
          <Alert variant="error" title="Hata">
            {error}
          </Alert>
        )}
        <div className="flex justify-end gap-3">
          <Button type="button" variant="secondary" onClick={onClose}>
            İptal
          </Button>
          <Button type="submit" isLoading={isPending} disabled={!namesText.trim()}>
            Toplu Oluştur
          </Button>
        </div>
      </form>
    </Modal>
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
