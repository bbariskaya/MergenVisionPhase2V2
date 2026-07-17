import { Card, CardContent } from '@/components/ui/Card'
import { ScanFace, UserPlus } from 'lucide-react'
import { Link } from 'react-router'

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <section className="rounded-2xl bg-gradient-to-br from-navy-900 via-navy-800 to-navy-900 p-6 text-white shadow-xl lg:p-8">
        <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">MergenVision Yüz Tanıma</h1>
        <p className="mt-2 max-w-xl text-sm text-navy-200 sm:text-base">
          Görseldeki yüzleri tanıyın, yeni kimlikler kaydedin ve işlem geçmişini görüntüleyin.
        </p>
      </section>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <Link to="/identify">
          <Card className="group h-full transition-shadow hover:shadow-md">
            <CardContent className="flex items-start gap-4 p-5">
              <div className="rounded-lg bg-primary-50 p-3 text-primary">
                <ScanFace className="h-6 w-6" />
              </div>
              <div>
                <h2 className="font-semibold text-navy-900 group-hover:text-primary">Yüz Tanıma</h2>
                <p className="mt-1 text-sm text-navy-500">Bir görsel yükleyip yüzleri tanıyın.</p>
              </div>
            </CardContent>
          </Card>
        </Link>

        <Card className="h-full">
          <CardContent className="flex items-start gap-4 p-5">
            <div className="rounded-lg bg-navy-100 p-3 text-navy-600">
              <UserPlus className="h-6 w-6" />
            </div>
            <div>
              <h2 className="font-semibold text-navy-900">Yeni Kişi Kaydı</h2>
              <p className="mt-1 text-sm text-navy-500">
                Kayıt, tanıma sonrası anonim yüzler için yapılır. Tanıma sonuçlarından “Kaydet” bağlantısını kullanın.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
