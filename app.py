import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Online Retail Executive Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- HELPER MEMUAT DATA (CACHED) ---
@st.cache_data
def load_data():
    # Membaca dengan path relatif untuk deployment mandiri / GitHub
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    # 1. Sales Data
    sales_path = os.path.join(data_dir, 'sales.parquet')
    df_sales = pd.read_parquet(sales_path) if os.path.exists(sales_path) else pd.DataFrame()
    if not df_sales.empty:
        df_sales['InvoiceDate'] = pd.to_datetime(df_sales['InvoiceDate'])
        df_sales['InvoiceMonth'] = df_sales['InvoiceDate'].dt.to_period('M').astype(str)

    # 2. RFM Segments
    rfm_path = os.path.join(data_dir, 'rfm_segments.csv')
    df_rfm = pd.read_csv(rfm_path) if os.path.exists(rfm_path) else pd.DataFrame()

    # 3. Association Rules
    rules_path = os.path.join(data_dir, 'rules.csv')
    df_rules = pd.read_csv(rules_path) if os.path.exists(rules_path) else pd.DataFrame()

    # 4. Rekomendasi Produk (CF, CB, Hybrid)
    rec_prod_path = os.path.join(data_dir, 'rekomendasi_produk.csv')
    df_rec_prod = pd.read_csv(rec_prod_path) if os.path.exists(rec_prod_path) else pd.DataFrame()

    # 5. Rekomendasi Pelanggan
    rec_cust_path = os.path.join(data_dir, 'rekomendasi_pelanggan.csv')
    df_rec_cust = pd.read_csv(rec_cust_path) if os.path.exists(rec_cust_path) else pd.DataFrame()

    # 6. Churn Scores
    churn_path = os.path.join(data_dir, 'churn_scores.csv')
    df_churn = pd.read_csv(churn_path) if os.path.exists(churn_path) else pd.DataFrame()

    # 7. Forecast Data
    forecast_path = os.path.join(data_dir, 'forecast.csv')
    df_forecast = pd.read_csv(forecast_path) if os.path.exists(forecast_path) else pd.DataFrame()

    return df_sales, df_rfm, df_rules, df_rec_prod, df_rec_cust, df_churn, df_forecast

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'churn_model.joblib')
    if os.path.exists(model_path):
        payload = joblib.load(model_path)
        return payload['model'], payload['features']
    return None, None

@st.cache_resource
def load_product_stats(_df_sales):
    # Pembeli tiap produk (pelanggan terdaftar) untuk menjelaskan skor CF,
    # dan jumlah invoice tiap produk untuk memilih produk default
    reg = _df_sales[_df_sales['CustomerID'].notna()]
    buyers = reg.groupby('Description')['CustomerID'].agg(lambda s: frozenset(s.unique())).to_dict()
    invoices = _df_sales.groupby('Description')['InvoiceNo'].nunique()
    return buyers, invoices

# Memuat semua aset
df_sales, df_rfm, df_rules, df_rec_prod, df_rec_cust, df_churn, df_forecast = load_data()
model_churn, model_features = load_model()

# --- SIDEBAR & NAVIGASI ---
st.sidebar.title("🧭 Navigasi Analitik")
daftar_halaman = [
    "📈 Ringkasan Penjualan",
    "👥 Segmen Pelanggan (RFM)",
    "🛒 Analisis Keranjang (MBA)",
    "🎁 Rekomendasi Produk",
    "🔮 Prediksi Churn Pelanggan",
]
# Halaman forecast hanya muncul jika data/forecast.csv tersedia (Tahap 8 bersifat opsional)
if not df_forecast.empty:
    daftar_halaman.append("📉 Forecast Pendapatan")
page = st.sidebar.radio("Pilih Halaman Dashboard:", daftar_halaman)

# ==========================================
# 1. HALAMAN RINGKASAN PENJUALAN
# ==========================================
if page == "📈 Ringkasan Penjualan":
    st.title("📈 Ringkasan Performa Penjualan")
    st.markdown("**Penjelasan untuk Awam:** Halaman ini menampilkan rangkuman indikator performa utama (KPI) ritel, tren perkembangan pendapatan kotor dari bulan ke bulan, konsentrasi penjualan antar negara, serta grafik kepadatan waktu tersibuk saat transaksi terjadi.")

    if df_sales.empty:
        st.warning("Data penjualan tidak ditemukan.")
    else:
        # Filter Negara (Multiselect)
        all_countries = sorted(df_sales['Country'].unique())
        selected_countries = st.sidebar.multiselect("Saring Negara (kosong = semua negara):", all_countries, default=[])

        # Filter Data
        if selected_countries:
            filtered_sales = df_sales[df_sales['Country'].isin(selected_countries)].copy()
        else:
            filtered_sales = df_sales.copy()

        # Hitung Metrik KPI
        total_revenue = filtered_sales['TotalPrice'].sum()
        total_invoices = filtered_sales['InvoiceNo'].nunique()
        total_customers = filtered_sales['CustomerID'].dropna().nunique()

        # Rata-rata/Median AOV
        invoice_revs = filtered_sales.groupby('InvoiceNo')['TotalPrice'].sum()
        median_aov = invoice_revs.median() if not invoice_revs.empty else 0

        # Layout KPI 4 Kolom
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Pendapatan (GBP)", f"£{total_revenue:,.2f}")
        kpi2.metric("Jumlah Transaksi (Invoice)", f"{total_invoices:,}")
        kpi3.metric("Pelanggan Unik (Terdaftar)", f"{total_customers:,}")
        kpi4.metric("Median Nilai Transaksi (AOV)", f"£{median_aov:,.2f}")

        st.markdown("--- ")

        col1, col2 = st.columns(2)

        with col1:
            # Tren Bulanan
            monthly_trend = filtered_sales.groupby('InvoiceMonth')['TotalPrice'].sum().reset_index()
            fig_trend = px.line(
                monthly_trend, x='InvoiceMonth', y='TotalPrice',
                title="Tren Pendapatan Bulanan (GBP £)",
                markers=True, labels={'TotalPrice': 'Total Pendapatan (£)', 'InvoiceMonth': 'Bulan'}
            )
            st.plotly_chart(fig_trend, width="stretch")
            st.caption("Catatan: data Desember 2011 hanya sampai 9 Desember, sehingga pendapatannya tampak turun tajam.")

        with col2:
            # Top 10 Produk
            top_products = filtered_sales.groupby('Description')['TotalPrice'].sum().reset_index()
            top_products = top_products.sort_values(by='TotalPrice', ascending=False).head(10)
            fig_prod = px.bar(
                top_products, x='TotalPrice', y='Description',
                orientation='h', title="Top 10 Produk Terlaris Berdasarkan Revenue (GBP £)",
                labels={'TotalPrice': 'Total Penjualan (£)', 'Description': 'Nama Produk'},
                color='TotalPrice', color_continuous_scale='Blues'
            )
            fig_prod.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_prod, width="stretch")
            # Catatan kualitas data: satu order besar yang kemudian dibatalkan lewat koreksi manual
            if ('PICNIC BASKET WICKER SMALL' in top_products['Description'].values
                    and (filtered_sales['InvoiceNo'].astype(str) == '556444').any()):
                st.caption("Catatan data: PICNIC BASKET WICKER SMALL masuk 10 besar karena satu invoice (556444, £38,970) "
                           "yang kemudian dibatalkan lewat koreksi manual. Tanpa invoice itu, produk ini tidak masuk 10 besar.")

        # Konsentrasi Penjualan per Negara
        st.subheader("🌍 Konsentrasi Penjualan per Negara")
        total_semua_negara = df_sales['TotalPrice'].sum()
        uk_share = df_sales.loc[df_sales['Country'] == 'United Kingdom', 'TotalPrice'].sum() / total_semua_negara * 100
        st.write(
            f"United Kingdom menyumbang **{uk_share:.2f}%** dari total pendapatan semua negara. "
            "Karena itu grafik menampilkan 10 negara teratas di luar UK agar perbedaannya terlihat. "
            "Tabel di sebelah kanan memuat semua negara; klik judul kolom untuk mengurutkan."
        )

        country_stats = filtered_sales.groupby('Country').agg(
            Pendapatan=('TotalPrice', 'sum'),
            Invoice=('InvoiceNo', 'nunique'),
            Pelanggan=('CustomerID', 'nunique')
        )
        country_stats['Median_Order'] = (
            filtered_sales.groupby(['Country', 'InvoiceNo'])['TotalPrice'].sum()
            .groupby(level='Country').median()
        )
        # Persentase selalu dihitung terhadap total semua negara (tidak ikut filter)
        country_stats['Persen_Pendapatan'] = country_stats['Pendapatan'] / total_semua_negara * 100
        country_stats = country_stats.sort_values(by='Pendapatan', ascending=False).reset_index()

        col3, col4 = st.columns(2)

        with col3:
            top_non_uk = country_stats[country_stats['Country'] != 'United Kingdom'].head(10)
            if top_non_uk.empty:
                st.info("Hanya United Kingdom yang dipilih, jadi grafik negara di luar UK tidak ditampilkan.")
            else:
                top_non_uk = top_non_uk.assign(Label=top_non_uk['Persen_Pendapatan'].map('{:.2f}%'.format))
                fig_country = px.bar(
                    top_non_uk, x='Pendapatan', y='Country', orientation='h', text='Label',
                    title="10 Negara Teratas di Luar UK (GBP £)",
                    labels={'Pendapatan': 'Total Pendapatan (£)', 'Country': 'Negara',
                            'Median_Order': 'Median Order (£)', 'Persen_Pendapatan': '% Pendapatan'},
                    hover_data={'Pendapatan': ':,.2f', 'Pelanggan': True, 'Invoice': True,
                                'Median_Order': ':,.2f', 'Persen_Pendapatan': ':.2f', 'Label': False}
                )
                fig_country.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_country, width="stretch")

        with col4:
            st.dataframe(
                country_stats,
                hide_index=True, width="stretch",
                column_order=['Country', 'Pendapatan', 'Persen_Pendapatan', 'Pelanggan', 'Median_Order', 'Invoice'],
                column_config={
                    'Country': st.column_config.TextColumn('Negara'),
                    'Pendapatan': st.column_config.NumberColumn('Pendapatan (£)', format='%,.2f'),
                    'Persen_Pendapatan': st.column_config.NumberColumn('% Pendapatan', format='%.2f%%'),
                    'Pelanggan': st.column_config.NumberColumn('Pelanggan', format='%,d'),
                    'Median_Order': st.column_config.NumberColumn('Median Order (£)', format='%,.2f'),
                    'Invoice': st.column_config.NumberColumn('Invoice', format='%,d'),
                }
            )

        # Heatmap Hari x Jam
        st.subheader("🗓️ Kepadatan Transaksi: Hari vs Jam")
        st.write("Analisis hari dan jam dengan frekuensi transaksi tertinggi untuk optimalisasi jadwal operasional gudang dan kampanye pemasaran digital.")

        filtered_sales['DayOfWeek'] = filtered_sales['InvoiceDate'].dt.day_name()
        filtered_sales['Hour'] = filtered_sales['InvoiceDate'].dt.hour
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Sunday']

        pivot_heat = filtered_sales.groupby(['DayOfWeek', 'Hour'])['InvoiceNo'].nunique().unstack().fillna(0)
        pivot_heat = pivot_heat.reindex(day_order)

        fig_heat = px.imshow(
            pivot_heat,
            labels=dict(x="Jam Transaksi (Hour)", y="Hari Pembelian", color="Jumlah Invoice"),
            x=pivot_heat.columns, y=pivot_heat.index,
            color_continuous_scale='YlGnBu', title="Heatmap Kepadatan Waktu Transaksi"
        )
        st.plotly_chart(fig_heat, width="stretch")

# ==========================================
# 2. HALAMAN SEGMEN PELANGGAN (RFM)
# ==========================================
elif page == "👥 Segmen Pelanggan (RFM)":
    st.title("👥 Segmentasi Perilaku Pelanggan (RFM + K-Means)")
    st.markdown("**Penjelasan untuk Awam:** Halaman ini memetakan pelanggan ke dalam 4 kelompok (segmen) berbeda berdasarkan kebiasaan belanja mereka: seberapa baru mereka belanja (**Recency**), seberapa sering (**Frequency**), dan berapa besar total uang yang dihabiskan (**Monetary**).")

    if df_rfm.empty:
        st.warning("Data segmen RFM tidak ditemukan.")
    else:
        # Profil segmen, dihitung dari rfm_segments.csv
        seg_profile = df_rfm.groupby('Segment').agg(
            Pelanggan=('CustomerID', 'count'),
            Total_Rev=('Monetary', 'sum'),
            Rata_Belanja=('Monetary', 'mean'),
            Median_Recency=('Recency', 'median'),
            Median_Frequency=('Frequency', 'median'),
            Median_Monetary=('Monetary', 'median'),
            Sekali_Belanja=('Frequency', lambda f: (f == 1).mean() * 100)
        ).reset_index()
        seg_profile['Persen_Pelanggan'] = seg_profile['Pelanggan'] / seg_profile['Pelanggan'].sum() * 100
        seg_profile['Persen_Revenue'] = seg_profile['Total_Rev'] / seg_profile['Total_Rev'].sum() * 100
        seg_profile = seg_profile.sort_values(by='Total_Rev', ascending=False)

        st.subheader("📊 Profil Karakteristik Segmen Pelanggan")
        st.dataframe(
            seg_profile, hide_index=True, width="stretch",
            column_order=['Segment', 'Pelanggan', 'Persen_Pelanggan', 'Persen_Revenue', 'Total_Rev', 'Rata_Belanja',
                          'Median_Recency', 'Median_Frequency', 'Median_Monetary', 'Sekali_Belanja'],
            column_config={
                'Segment': st.column_config.TextColumn('Segmen'),
                'Pelanggan': st.column_config.NumberColumn('Pelanggan', format='%,d'),
                'Persen_Pelanggan': st.column_config.NumberColumn('% Pelanggan', format='%.2f%%'),
                'Persen_Revenue': st.column_config.NumberColumn('% Revenue', format='%.2f%%'),
                'Total_Rev': st.column_config.NumberColumn('Total Belanja (£)', format='%,.2f'),
                'Rata_Belanja': st.column_config.NumberColumn('Rata-rata Belanja (£)', format='%,.2f'),
                'Median_Recency': st.column_config.NumberColumn('Median Recency (hari)', format='%.0f'),
                'Median_Frequency': st.column_config.NumberColumn('Median Frequency', format='%.0f'),
                'Median_Monetary': st.column_config.NumberColumn('Median Monetary (£)', format='%,.2f'),
                'Sekali_Belanja': st.column_config.NumberColumn('% Sekali Belanja', format='%.1f%%',
                    help='Persentase pelanggan di segmen ini yang baru bertransaksi satu kali'),
            }
        )
        st.caption("% Revenue dihitung terhadap total belanja pelanggan terdaftar, bukan terhadap total pendapatan di halaman Ringkasan.")

        # Sorotan otomatis dari data
        top_seg = seg_profile.iloc[0]
        sorotan = [f"**{top_seg['Segment']}** hanya {top_seg['Persen_Pelanggan']:.2f}% pelanggan, tetapi menyumbang "
                   f"{top_seg['Persen_Revenue']:.2f}% total belanja."]
        loyal = df_rfm[df_rfm['Segment'] == 'Loyal']
        if not loyal.empty:
            loyal_jauh = loyal[loyal['Recency'] > 90]
            sorotan.append(f"**{len(loyal_jauh):,} pelanggan Loyal** ({len(loyal_jauh) / len(loyal) * 100:.1f}%) sudah lebih dari "
                           f"90 hari tidak belanja, dengan total belanja £{loyal_jauh['Monetary'].sum():,.0f}. "
                           "Mereka target utama kampanye retensi.")
        st.markdown("**Sorotan:**\n" + "\n".join(f"- {teks}" for teks in sorotan))

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📖 Arti Tiap Segmen")
            st.markdown("""
            *   **Champions:** Pelanggan VIP Anda yang belanja sangat sering, bernominal besar, dan bertransaksi baru-baru ini.
            *   **Loyal:** Pelanggan andalan yang bertransaksi stabil dan rutin kembali belanja.
            *   **New/Promising:** Pelanggan baru dengan frekuensi transaksi sedikit namun potensial ditingkatkan.
            *   **Hibernating:** Kelompok pelanggan pasif berisiko tinggi yang sudah sangat lama tidak kembali bertransaksi.
            """)

        with col2:
            st.subheader("🎯 Sebaran Pelanggan (Recency vs Monetary)")
            fig_scatter = px.scatter(
                df_rfm, x='Recency', y='Monetary', color='Segment',
                log_x=True, log_y=True,
                title="Peta Distribusi Pelanggan pada Skala Logaritma",
                labels={'Recency': 'Hari Sejak Belanja Terakhir (Recency)', 'Monetary': 'Total Pengeluaran Belanja (Monetary)'},
                color_discrete_sequence=px.colors.qualitative.Set1,
                hover_data=['CustomerID']
            )
            st.plotly_chart(fig_scatter, width="stretch")

# ==========================================
# 3. HALAMAN MARKET BASKET ANALYSIS (MBA)
# ==========================================
elif page == "🛒 Analisis Keranjang (MBA)":
    st.title("🛒 Analisis Pola Keranjang Belanja (Association Rules)")
    st.markdown("**Penjelasan untuk Awam:** Halaman ini mendeteksi pola kombinasi produk yang paling sering dibeli secara bersamaan oleh pelanggan di dalam satu transaksi menggunakan algoritma Apriori.")

    if df_rules.empty:
        st.warning("Data aturan asosiasi tidak ditemukan.")
    else:
        # Slider Filter
        st.sidebar.subheader("Saring Aturan Asosiasi")
        min_conf = st.sidebar.slider("Minimum Confidence:", 0.1, 1.0, 0.5, 0.05)
        min_lift = st.sidebar.slider("Minimum Lift (Kekuatan Hubungan):", 1.0, 30.0, 1.0, 0.5)

        # Jumlah invoice di balik support, confidence, dan lift.
        # Apriori dihitung dari invoice pelanggan terdaftar (yang punya CustomerID).
        n_invoice = df_sales.loc[df_sales['CustomerID'].notna(), 'InvoiceNo'].nunique()
        rules_view = df_rules.copy()
        rules_view['Invoice A+B'] = (rules_view['support'] * n_invoice).round().astype(int)
        rules_view['Invoice A'] = (rules_view['Invoice A+B'] / rules_view['confidence']).round().astype(int)
        rules_view['Invoice B'] = (rules_view['confidence'] / rules_view['lift'] * n_invoice).round().astype(int)

        kolom_tampil = ['antecedents', 'consequents', 'support', 'confidence', 'lift',
                        'Invoice A+B', 'Invoice A', 'Invoice B']
        konfigurasi_kolom = {
            'antecedents': st.column_config.TextColumn('A · antecedents (JIKA membeli)', width=300),
            'consequents': st.column_config.TextColumn('B · consequents (MAKA juga membeli)', width=300),
            'support': st.column_config.NumberColumn('support', format='%.4f', width=80,
                help='Invoice A+B ÷ total invoice'),
            'confidence': st.column_config.NumberColumn('confidence', format='%.4f', width=95,
                help='Invoice A+B ÷ Invoice A'),
            'lift': st.column_config.NumberColumn('lift', format='%.2f', width=70,
                help='confidence ÷ (Invoice B ÷ total invoice)'),
            'Invoice A+B': st.column_config.NumberColumn('Invoice A+B', format='%,d', width=95,
                help='Jumlah invoice yang memuat A dan B sekaligus'),
            'Invoice A': st.column_config.NumberColumn('Invoice A', format='%,d', width=85,
                help='Jumlah invoice yang memuat A'),
            'Invoice B': st.column_config.NumberColumn('Invoice B', format='%,d', width=85,
                help='Jumlah invoice yang memuat B'),
        }

        # Filter Aturan
        df_rules_filtered = rules_view[
            (rules_view['confidence'] >= min_conf) &
            (rules_view['lift'] >= min_lift)
        ].sort_values(by='lift', ascending=False)

        st.write(f"Dihitung dari **{n_invoice:,}** invoice pelanggan terdaftar. "
                 f"Ditemukan **{len(df_rules_filtered)}** aturan asosiasi kuat yang memenuhi kriteria filter Anda.")

        # Tampilkan tabel utama
        st.dataframe(
            df_rules_filtered[kolom_tampil],
            width="stretch", hide_index=True, column_config=konfigurasi_kolom
        )
        st.caption(
            f"Cara menghitung: support = Invoice A+B ÷ {n_invoice:,} · "
            "confidence = Invoice A+B ÷ Invoice A · "
            f"lift = confidence ÷ (Invoice B ÷ {n_invoice:,})"
        )

        st.markdown("--- ")
        st.subheader("🔍 Cari Rekomendasi Asosiasi per Produk Spesifik")
        st.write("Pilih produk untuk melihat barang apa saja yang paling berpeluang dibeli bersama produk tersebut. "
                 "Daftar ini mengikuti filter confidence dan lift di sidebar.")

        # List unik produk dari antecedents pada aturan yang lolos filter
        all_antes = sorted(df_rules_filtered['antecedents'].unique().tolist())
        if not all_antes:
            st.info("Tidak ada aturan yang lolos filter saat ini. Turunkan Minimum Confidence atau Minimum Lift di sidebar.")
        else:
            selected_prod = st.selectbox("Pilih Produk Utama:", all_antes)
            match_rules = df_rules_filtered[df_rules_filtered['antecedents'] == selected_prod]
            st.dataframe(
                match_rules[kolom_tampil],
                width="stretch", hide_index=True, column_config=konfigurasi_kolom
            )

# ==========================================
# 4. HALAMAN REKOMENDASI PRODUK (CF, CB, HYBRID)
# ==========================================
elif page == "🎁 Rekomendasi Produk":
    st.title("🎁 Sistem Rekomendasi Pintar Lini Produk")
    st.markdown("**Penjelasan untuk Awam:** Pilih sebuah nama barang untuk melihat hasil rekomendasi otomatis menggunakan tiga algoritma pemodelan canggih secara berdampingan.")

    if df_rec_prod.empty:
        st.warning("Data rekomendasi produk tidak ditemukan.")
    else:
        all_unique_prods = sorted(df_rec_prod['Produk'].unique().tolist())
        buyers, invoices_per_prod = load_product_stats(df_sales)

        # Default: produk yang paling sering dibeli (jumlah invoice terbanyak)
        kandidat = invoices_per_prod[invoices_per_prod.index.isin(all_unique_prods)]
        default_prod = kandidat.idxmax() if not kandidat.empty else all_unique_prods[0]
        selected_prod = st.selectbox("Pilih Nama Produk yang Ingin Direferensikan:", all_unique_prods,
                                     index=all_unique_prods.index(default_prod))
        st.caption(f"Pilihan awal: {default_prod}, produk yang paling sering dibeli. Ketik sebagian nama untuk mencari produk lain.")

        if selected_prod:
            prod_df = df_rec_prod[df_rec_prod['Produk'] == selected_prod]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("🔗 Collaborative Filtering")
                st.write("*Berdasarkan produk yang paling sering dibeli bersama oleh orang lain.*")
                cf_df = prod_df[prod_df['Metode'] == 'Collaborative Filtering'].sort_values(by='Rank')
                if not cf_df.empty:
                    st.dataframe(cf_df[['Rank', 'Rekomendasi', 'Skor']], width="stretch", hide_index=True)
                else:
                    st.info("Rekomendasi CF kosong.")

            with col2:
                st.subheader("📝 Content-Based Filtering")
                st.write("*Berdasarkan kemiripan arti kosakata teks nama deskripsi produk.*")
                cb_df = prod_df[prod_df['Metode'] == 'Content-Based Filtering'].sort_values(by='Rank')
                if not cb_df.empty:
                    st.dataframe(cb_df[['Rank', 'Rekomendasi', 'Skor']], width="stretch", hide_index=True)
                else:
                    st.info("Rekomendasi CB kosong.")

            with col3:
                st.subheader("🧬 Hybrid System")
                st.write("*Kombinasi terbaik antara kesamaan teks nama dan kesamaan perilaku belanja.*")
                hy_df = prod_df[prod_df['Metode'] == 'Hybrid'].sort_values(by='Rank')
                if not hy_df.empty:
                    st.dataframe(hy_df[['Rank', 'Rekomendasi', 'Skor']], width="stretch", hide_index=True)
                else:
                    st.info("Rekomendasi Hybrid kosong.")

            st.caption("Skor hanya bisa dibandingkan di dalam kolom yang sama; skala skor ketiga metode berbeda.")

            # Rincian hitungan skor CF: berapa pelanggan membeli kedua produk
            if not cf_df.empty and selected_prod in buyers:
                with st.expander("🔎 Lihat hitungan skor Collaborative Filtering (jumlah pembeli)"):
                    pembeli_terpilih = buyers[selected_prod]
                    rincian = cf_df[['Rank', 'Rekomendasi', 'Skor']].copy()
                    rincian['Pembeli Keduanya'] = rincian['Rekomendasi'].map(
                        lambda p: len(pembeli_terpilih & buyers.get(p, frozenset())))
                    rincian['Pembeli Produk Ini'] = rincian['Rekomendasi'].map(lambda p: len(buyers.get(p, frozenset())))
                    st.write(f"**{selected_prod}** dibeli oleh **{len(pembeli_terpilih):,}** pelanggan terdaftar.")
                    st.dataframe(
                        rincian[['Rank', 'Rekomendasi', 'Pembeli Keduanya', 'Pembeli Produk Ini', 'Skor']],
                        width="stretch", hide_index=True,
                        column_config={
                            'Pembeli Keduanya': st.column_config.NumberColumn(
                                'Pembeli Keduanya', format='%,d',
                                help='Jumlah pelanggan yang membeli produk terpilih dan produk rekomendasi'),
                            'Pembeli Produk Ini': st.column_config.NumberColumn(
                                'Pembeli Produk Rekomendasi', format='%,d'),
                            'Skor': st.column_config.NumberColumn('Skor CF', format='%.4f'),
                        }
                    )
                    contoh = rincian.iloc[0]
                    st.caption(
                        "Skor CF = Pembeli Keduanya ÷ √(pembeli produk terpilih × pembeli produk rekomendasi). "
                        f"Contoh peringkat 1: {contoh['Pembeli Keduanya']:,} ÷ √({len(pembeli_terpilih):,} × "
                        f"{contoh['Pembeli Produk Ini']:,}) = {contoh['Skor']:.4f}"
                    )

# ==========================================
# 5. HALAMAN PREDIKSI CHURN PELANGGAN
# ==========================================
elif page == "🔮 Prediksi Churn Pelanggan":
    st.title("🔮 Analisis Risiko Churn Pelanggan")
    st.markdown("**Penjelasan untuk Awam:** Churn berarti pelanggan berhenti berbelanja di toko kita. Halaman ini mendeteksi peluang (probabilitas) pelanggan tertentu untuk pergi agar tim pemasaran dapat segera memberikan promo penyelamatan sebelum terlambat.")

    if df_churn.empty or df_rec_cust.empty or model_churn is None:
        st.warning("Model churn atau tabel probabilitas pendukung tidak berhasil dimuat.")
    else:
        # Dropdown CustomerID
        all_custs = sorted(df_churn['CustomerID'].unique().tolist())
        selected_cust = st.selectbox("Pilih ID Pelanggan (CustomerID) yang Ingin Diprediksi:", all_custs)

        if selected_cust:
            cust_row = df_churn[df_churn['CustomerID'] == selected_cust].iloc[0]

            # Tampilkan Ringkasan Informasi Pelanggan Saat Ini
            st.subheader("ℹ️ Profil Belanja Aktif Saat Ini")
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Segmen RFM", f"{cust_row['Segment']}")
            col2.metric("Probabilitas Churn", f"{cust_row['Churn_Prob']*100:.2f}%")
            col3.metric("Pengeluaran Historis", f"£{cust_row['Monetary']:,.2f}")
            col4.metric("Hari Sejak Belanja Terakhir", f"{int(cust_row['Recency'])} hari")

            # Rekomendasi Personal
            st.subheader(f"🎁 Rekomendasi Produk Personal untuk Pelanggan {selected_cust}")
            st.write("Produk berikut disesuaikan secara khusus sesuai dengan selera historis profil belanja pelanggan.")
            cust_rec_df = df_rec_cust[df_rec_cust['CustomerID'] == selected_cust].sort_values(by='Rank')
            if not cust_rec_df.empty:
                st.dataframe(cust_rec_df[['Rank', 'Rekomendasi', 'Skor']], width="stretch", hide_index=True)
                st.caption("Skor = jumlah kemiripan (cosine similarity) dari 10 pelanggan dengan pola belanja paling mirip "
                           "yang membeli produk tersebut. Maksimum 10, bukan persen; makin tinggi, makin banyak pelanggan "
                           "serupa yang membelinya.")
            else:
                st.info("Tidak ada riwayat rekomendasi personal untuk pelanggan ini.")

            # Expander Simulasi What-If
            st.markdown("--- ")
            with st.expander("🧪 Simulasi Skenario What-If (Bagaimana jika perilakunya berubah?)"):
                st.write("Ubah nilai parameter fitur perilaku pelanggan di bawah ini untuk melihat simulasi dampak perubahan probabilitas Churn secara langsung dari mesin model Machine Learning.")

                # Buat input form sesuai urutan features model
                col_s1, col_s2, col_s3 = st.columns(3)

                with col_s1:
                    sim_freq = int(st.number_input("Jumlah Transaksi Historis (Frequency):", min_value=1, value=int(cust_row['Frequency'])))
                    sim_mon = float(st.number_input("Total Pengeluaran (Monetary):", min_value=1.0, value=float(cust_row['Monetary'])))
                    sim_qty = int(st.number_input("Total Unit Kuantitas (TotalQuantity):", value=int(cust_row['TotalQuantity'])))
                    sim_uniq = int(st.number_input("Jumlah Jenis Produk Berbeda (UniqueProducts):", min_value=1, value=int(cust_row['UniqueProducts'])))

                with col_s2:
                    sim_rec = int(st.number_input("Hari Sejak Transaksi Terakhir (Recency):", min_value=0, value=int(cust_row['Recency'])))
                    sim_tenure = int(st.number_input("Rentang Hari Aktif Sejak Belanja Pertama (Tenure):", min_value=0, value=int(cust_row['Tenure'])))
                    sim_gap = float(st.number_input("Rata-rata Selisih Hari Belanja (AvgDaysBetween):", min_value=0.0, value=float(cust_row['AvgDaysBetween'])))

                with col_s3:
                    sim_aov = float(st.number_input("Rata-rata Nilai per Order (AvgOrderValue):", min_value=0.0, value=float(cust_row['AvgOrderValue'])))
                    sim_last90 = int(st.number_input("Jumlah Transaksi dalam 90 Hari Terakhir (Freq_last90):", min_value=0, value=int(cust_row['Freq_last90'])))
                    sim_is_uk = int(st.selectbox("Apakah Berdomisili di UK? (IsUK: 1=Ya, 0=Tidak):", [1, 0], index=0 if int(cust_row['IsUK']) == 1 else 1))

                # Lakukan prediksi ulang
                input_data = pd.DataFrame([[
                    sim_freq, sim_mon, sim_uniq, sim_qty, sim_rec, sim_tenure, sim_aov, sim_gap, sim_last90, sim_is_uk
                ]], columns=model_features)

                new_prob = model_churn.predict_proba(input_data)[0, 1]
                new_pred = "🚨 CHURN" if new_prob >= 0.5 else "✅ TETAP AKTIF"

                st.markdown("--- ")
                st.markdown(f"### **Hasil Simulasi Prediksi Baru:**")
                st.subheader(f"Probabilitas Churn Baru: **{new_prob*100:.2f}%** ({new_pred})")

        # Tabel 20 Pelanggan Bernilai Tinggi dengan Risiko Churn Terbesar
        st.markdown("--- ")
        st.subheader("⚠️ Daftar Prioritas Tindakan: Top 20 Pelanggan Bernilai Tinggi di Bawah Ancaman Churn")
        st.write("Tabel ini mengidentifikasi kelompok pelanggan paling berharga (Monetary di atas persentil ke-80) yang memiliki risiko churn terbesar berdasarkan analisis model.")

        mon_80_pct = df_churn['Monetary'].quantile(0.80)
        val_risk_customers = df_churn[df_churn['Monetary'] >= mon_80_pct].sort_values(by='Churn_Prob', ascending=False).head(20)

        tabel_prioritas = val_risk_customers[['CustomerID', 'Segment', 'Monetary', 'Recency', 'Frequency', 'Churn_Prob']].copy()
        tabel_prioritas['Churn_Prob'] = tabel_prioritas['Churn_Prob'] * 100
        st.dataframe(
            tabel_prioritas, width="stretch", hide_index=True,
            column_config={
                'CustomerID': st.column_config.NumberColumn('CustomerID', format='%d'),
                'Segment': st.column_config.TextColumn('Segmen RFM'),
                'Monetary': st.column_config.NumberColumn('Total Belanja (£)', format='%,.2f'),
                'Recency': st.column_config.NumberColumn('Hari Sejak Belanja Terakhir', format='%d'),
                'Frequency': st.column_config.NumberColumn('Jumlah Transaksi', format='%d'),
                'Churn_Prob': st.column_config.NumberColumn('Probabilitas Churn', format='%.2f%%'),
            }
        )
        st.caption(f"Batas pelanggan bernilai tinggi: total belanja minimal £{mon_80_pct:,.2f} (persentil ke-80). "
                   "Diurutkan dari probabilitas churn tertinggi.")
        # Catatan kualitas data: order yang dibatalkan lewat koreksi manual masih terhitung
        if (15098 in val_risk_customers['CustomerID'].values
                and (df_sales['InvoiceNo'].astype(str) == '556444').any()):
            st.caption("Catatan data: total belanja CustomerID 15098 termasuk order £38,970 (invoice 556444) yang "
                       "dibatalkan lewat koreksi manual; belanja bersihnya hanya £649.50.")

# ==========================================
# 6. HALAMAN FORECAST PENDAPATAN
# ==========================================
elif page == "📉 Forecast Pendapatan":
    st.title("📉 Proyeksi Pendapatan Mingguan")
    st.markdown("**Penjelasan untuk Awam:** Halaman ini menampilkan hasil prediksi atau peramalan (*forecasting*) tren pendapatan ritel di masa mendatang berdasarkan pola transaksi historis.")

    if df_forecast.empty:
        st.info("Visualisasi modul peramalan tidak aktif karena berkas data/forecast.csv tidak tersedia di memori penyimpanan.")
    else:
        st.subheader("📈 Kurva Perbandingan Data Aktual & Proyeksi Forecast")
        # Format data/forecast.csv: kolom Minggu, Revenue, Jenis ('Aktual' atau 'Forecast')
        fig_fore = px.line(
            df_forecast, x='Minggu', y='Revenue', color='Jenis',
            title="Pendapatan Mingguan: Aktual vs Forecast",
            markers=True, labels={'Revenue': 'Pendapatan (GBP £)', 'Minggu': 'Minggu'}
        )
        st.plotly_chart(fig_fore, width="stretch")
