"""Malaysian Malay text normalizer with canonical-variant correction."""

from __future__ import annotations
import re
from .basic import BasicTextNormalizer

# Maps variant spellings → single canonical form.
# Covers the top substitution pairs found in error analysis of evaluation data.
_CANONICAL: dict[str, str] = {
    # okay cluster
    'okay': 'ok',
    'okey': 'ok',
    'oke':  'ok',

    # colloquial shortenings → full form
    'tu':    'itu',
    'ni':    'ini',
    'nih':   'ini',   # colloquial variant of ni/ini
    'jugak': 'juga',
    'pulak': 'pula',
    'bawak': 'bawa',
    'ku':    'aku',

    # second-person pronoun variants → shorter spoken form
    'engkau': 'kau',

    # yeah/yes → ya
    'yeah': 'ya',
    'yes':  'ya',
    'yep':  'ya',
    'yah':  'ya',

    # nope → no
    'nope': 'no',

    # formal ↔ informal alternation (→ shorter / spoken form)
    'ataupun': 'atau',
    'mahupun': 'maupun',
    'tetapi':  'tapi',
    'sahaja':  'je',     # formal sahaja → colloquial je (same as saja → je)
    'baharu':  'baru',
    'kecik':   'kecil',
    'aje':     'je',     # colloquial variant of je/saja

    # Malaysian vs Indonesian spelling (→ Malaysian standard)
    'yaitu':       'iaitu',
    'hewan':       'haiwan',
    'standar':     'standard',
    'film':        'filem',
    'project':     'projek',
    'business':    'bisnes',
    'kesehatan':   'kesihatan',
    'berbagai':    'pelbagai',
    'tentara':     'tentera',
    'sehat':       'sihat',
    'kawin':       'kahwin',
    'scammer':     'skamer',
    'nomor':       'nombor',
    'karena':      'kerana',
    'coba':        'cuba',
    'keuangan':    'kewangan',
    'modern':      'moden',
    'populer':     'popular',
    'muzik':       'muzik',   # already canonical; Indonesian 'musik' → Malaysian 'muzik'
    'musik':       'muzik',
    'ekspor':      'eksport',
    'eksport':     'eksport',
    'selingkoh':   'selingkuh',
    'lif':         'lift',
    'menyulu':     'menyuluh',

    # English loanwords → Malaysian Malay standard spelling
    'station':     'stesen',
    'cyclone':     'siklon',
    'parasites':   'parasit',
    'parasite':    'parasit',
    'krew':        'kru',      # crew (English krew → Malay kru)
    'garison':     'garrison', # spelling variant (single r)
    'uighur':      'uyghur',   # romanization variant
    'hello':       'helo',
    'suit':        'sut',
    'point':       'poin',
    'experiment':  'eksperimen',
    'exotic':      'eksotik',
    'audience':    'audiens',
    'variant':     'varian',
    'majority':    'majoriti',
    'card':        'kad',

    # colloquial spoken variants → standard form
    'mintak':      'minta',    # colloquial of minta (please/request)
    'amek':        'ambil',    # colloquial of ambil (to take)
    'aja':         'je',       # Indonesian colloquial of saja → je
    'sri':         'seri',     # spelling variant of Malay honorific Seri
    'campakan':    'campakkan', # spelling variant (-kan suffix)
    'kemana':      'ke mana',  # fused form of ke mana (to where)
    'carrom':      'karom',    # English spelling of carrom game
    'karam':       'karom',    # phonetic variant of carrom/karom

    # fused compounds → space-separated canonical (split by _canonicalize)
    'bagitahu':    'bagi tahu',  # bagi tahu = to inform/tell
    'takpe':       'tak apa',    # takpe = tak apa = never mind
    'takpa':       'tak apa',    # variant spelling of takpe

    # ku- verb prefix fused forms (ku = aku as actor prefix)
    # Add more as encountered; a general rule risks splitting kurang, kuliah, etc.
    'kusanjungi':  'aku sanjungi',

    # colloquial pronoun variants
    'dorang': 'diorang',   # dorang = diorang = they/them (colloquial)

    # colloquial negation variants → tiada
    'takde':  'tiada',
    'takdak': 'tiada',
    'takda':  'tiada',

    # sekejap / kejap → jap  (canonical short form; both map to same target)
    # NOTE: do NOT add a second 'sekejap' key here — Python dicts keep the last
    # value, so a duplicate key silently overrides the first.
    'sekejap': 'jap',
    'kejap':   'jap',

    # pakcik written as one word → two tokens (space handled by _canonicalize)
    'pakcik': 'pak cik',

    'seorang': 'sorang',
    'alah':    'ala',

    # honorific
    'dato': 'datuk',

    # medical spelling: ensefalitis is the Malaysian standard
    'encefalitis':  'ensefalitis',
    'encephalitis': 'ensefalitis',

    # budget: Malaysian standard is 'bajet'; English 'budget' → bajet
    'budget': 'bajet',

    # greeting
    'hi': 'hai',

    # filler normalisation
    'aa':  'ah',
    'ha':  'ah',
    'oh':  'ah',
    'uh':  'ah',
    'err': 'ah',
    'haa': 'ah',
    'aah': 'ah',
    'mm':  'hmm',
    'hm':  'hmm',
    'ehm': 'um',
    'umm': 'um',

    # English/Malay spelling variants
    'minute':  'minit',
    'minutes': 'minit',

    # tau / tahu — colloquial vs standard
    'tau': 'tahu',

    # loanword spelling
    'chancellor': 'canselor',

    # deklalasi — phonetic r→l model error; map TO the correct spelling
    'deklalasi': 'deklarasi',

    # bahawa/bahwa — Indonesian vs Malaysian standard spelling of the conjunction
    'bahwa': 'bahawa',

    # saspek — dialectal pronunciation of suspek (loanword from "suspect")
    'saspek': 'suspek',

    # proper name spelling variants
    'syaril':  'syahril',
    'sharil':  'syahril',
    'shahril': 'syahril',
    'sharyl':  'syahril',

    # fuiyoh cluster — Malaysian exclamation of amazement
    'fuyuh':  'fuiyoh',
    'fuyoh':  'fuiyoh',
    'fuyo':   'fuiyoh',

    # name variants
    'ama':    'amar',
    'ting':   'teng',   # teng teng ↔ ting ting (onomatopoeia / name variant)
    'amma':   'amar',
    'alayas': 'alias',
    'yusof':  'yusuf',
    'nasya':  'nasha',
    'nasyah': 'nasha',
    'tishen': 'tishan',
    'tisyen': 'tishan',
    'tisyan': 'tishan',
    'sula':   'sulah',

    # sabok/sabut → sabuk (dialectal spelling variants)
    'sabok':  'sabuk',
    'sabut':  'sabuk',

    # apasal → pasal (colloquial "why" → root word)
    'apasal': 'pasal',

    # committee: English spellings → Malay loanword
    'committee':  'komiti',
    'commitee':   'komiti',   # common misspelling

    # ─── Indonesian → Malaysian standard (batch from error analysis) ───
    'inggris':          'inggeris',    # English (language)
    'prancis':          'perancis',    # France / French
    'klub':             'kelab',       # club
    'sop':              'sup',         # soup
    'mie':              'mi',          # noodle
    'mulai':            'mula',        # begin (Indonesian mulai → Malaysian mula)
    'rekor':            'rekod',       # record
    'pribadi':          'peribadi',    # personal/private
    'dokter':           'doktor',      # doctor
    'negri':            'negeri',      # state (spelling variant)
    'pengantar':        'penghantar',  # sender/deliverer (hantar vs antar)
    'diajukan':         'diacukan',    # proposed/submitted
    'menyehatkan':      'menyihatkan', # to make healthy (sehat → sihat)

    # ─── English → Malaysian Malay spelling ───
    'rose':             'ros',         # the flower
    'agent':            'ejen',        # agent
    'agen':             'ejen',        # agent (Indonesian loanword form)
    'kontrol':          'control',     # control (references use English spelling)
    'size':             'saiz',        # size
    'canada':           'kanada',      # Canada
    'service':          'servis',      # service
    'case':             'kes',         # case
    'episode':          'episod',      # episode
    'hobby':            'hobi',        # hobby
    'halo':             'helo',        # hello/greeting (halo = English, helo = Malay)
    'informative':      'informatif',  # informative
    'rome':             'rom',         # Rome
    'china':            'cina',        # China
    'makkah':           'mekah',       # Mecca (Arabic romanization → Malay)
    'stephen':          'steven',      # name variant
    'highlands':        'highland',    # singular vs plural (Cameron Highland)

    # ─── Spelling variants / phonetic model errors ───
    'alayarham':        'allahyarham', # Arabic honorific (phonetic variant)
    'alaiyarham':       'allahyarham',
    'macik':            'makcik',      # aunt/madam (colloquial shortening)
    'macic':            'makcik',
    'dajal':            'dajjal',      # Antichrist (Arabic: الدجال)
    'mohamad':          'muhammad',    # name romanization variants
    'mohammad':         'muhammad',
    'mohamed':          'muhammad',
    'muhamad':          'muhammad',   # missing double-m
    'mohd':             'muhammad',   # common abbreviation
    'ramli':            'ramlee',      # P. Ramlee (Malaysian cultural icon)
    'saja':             'je',          # only/just (formal saja ↔ colloquial je)
    'cemelang':         'cemerlang',   # excellent (missing r)
    'menihatkan':       'menyihatkan', # to make healthy (ny → n model error)
    'dianugahkan':      'dianugerahkan', # awarded (missing letters)
    'musiba':           'musibah',     # calamity (truncated)
    'sud':              'sut',         # suit (phonetic variant)
    'moto':             'motor',       # motorcycle (truncated)
    'ambik':            'ambil',       # to take (colloquial variant of ambil/amek)
    'khaled':           'khalid',      # name romanization variant
    'gegeran':          'gegaran',     # tremor/vibration (vowel swap)
    'enfalitis':        'ensefalitis', # encephalitis (missing syllable)
    'talipon':          'telefon',     # telephone (dialectal variant)
    'pengekstrakkan':   'pengekstrakan', # extraction (double k)
    'basco':            'vasco',       # Vasco (da Gama) phonetic variant
    'pekara':           'perkara',     # matter/issue (missing r)
    'penjaman':         'pinjaman',    # loan (phonetic variant)
    'pesyarah':         'pensyarah',   # lecturer (missing n)
    'persyarah':        'pensyarah',   # lecturer (extra r)
    'gule':             'gula',        # sugar (Javanese/dialectal)

    # ─── Second-sweep additions ───
    # Indonesian → Malaysian
    'jumat':            'jumaat',      # Friday
    'resmi':            'rasmi',       # official
    'kendaraan':        'kenderaan',   # vehicle
    'menelpon':         'menelefon',   # to telephone
    # English → Malay
    'mummy':            'mami',        # mum/mother
    # Spelling variants
    'peniagaan':        'perniagaan',  # business (missing r)
    'aska':             'askar',       # soldier (truncated)
    'ditutor':          'ditutur',     # spoken/uttered (vowel swap)
    'siminov':          'siminoff',    # proper name romanization variant

    # ─── Third-sweep additions (aisyah-pro error analysis) ───
    # coffee/kopi — English loanword vs Malay native word; both in common use
    'coffee':           'kopi',
    # email/emel — "emel" is the Malaysian standard Malay form
    'email':            'emel',
    # assistant/asisten — "asisten" is the Malaysian Malay loanword
    'assistant':        'asisten',
    # dentist/dentis — "dentis" is the Malaysian Malay loanword
    'dentist':          'dentis',
    # maldives — singular form is "maldive" in reference transcripts
    'maldives':         'maldive',

    # ─── Fourth-sweep additions (dataset-wide word-confusion scan) ───
    # di + word fused phrasal forms (di = preposition "at/in", not the
    # passive verb prefix — that case is always fused and left alone).
    'dimana':   'di mana',    # where
    'disana':   'di sana',    # there
    'diantara': 'di antara',  # among/between

    # mu = contraction of kamu (your/you), same pattern as ku -> aku
    'mu': 'kamu',

    # ─── Reduplicated words fused without a hyphen (kanak-kanak -> kanakkanak) ───
    # Malay reduplication (word-word) marks plural/repetition/variety; models
    # frequently drop the hyphen. Canonical form splits back into two tokens.
    'kanakkanak':      'kanak kanak',
    'apaapa':          'apa apa',
    'anakanak':        'anak anak',
    'kirakira':        'kira kira',
    'undangundang':    'undang undang',
    'manamana':        'mana mana',
    'negaranegara':    'negara negara',
    'orangorang':      'orang orang',
    'kadangkadang':    'kadang kadang',
    'kawankawan':      'kawan kawan',
    'mulamula':        'mula mula',
    'tempattempat':    'tempat tempat',
    'kucingkucing':    'kucing kucing',
    'haiwanhaiwan':    'haiwan haiwan',
    'rakanrakan':      'rakan rakan',
    'nilainilai':      'nilai nilai',
    'filemfilem':      'filem filem',
    'makananmakanan':  'makanan makanan',
    'ahliahli':        'ahli ahli',
    'masingmasing':    'masing masing',
    'detikdetik':      'detik detik',
    'aktivitiaktiviti': 'aktiviti aktiviti',
    'lainlain':        'lain lain',
    'memorimemori':    'memori memori',
    'perkaraperkara':  'perkara perkara',
    'ceritacerita':    'cerita cerita',
    'golongangolongan': 'golongan golongan',
    'risikorisiko':    'risiko risiko',
    'pelajarpelajar':  'pelajar pelajar',
    'bangunanbangunan': 'bangunan bangunan',
    'bungabunga':      'bunga bunga',
    'harihari':        'hari hari',
    'ayatayat':        'ayat ayat',
    'pandangpandang':  'pandang pandang',
    'selsel':          'sel sel',
    'budakbudak':      'budak budak',
    'macammacam':      'macam macam',
    'kapalkapal':      'kapal kapal',
    'kerjakerja':      'kerja kerja',
    'pegawaipegawai':  'pegawai pegawai',
    'lubanglubang':    'lubang lubang',
    'tengahtengah':    'tengah tengah',
    'hatihati':        'hati hati',
    'barubaru':        'baru baru',
    'sekolahsekolah':  'sekolah sekolah',
    'bendabenda':      'benda benda',
    'puanpuan':        'puan puan',
    'jalanjalan':      'jalan jalan',
    'herohero':        'hero hero',
    'baikbaik':        'baik baik',
    'sebabsebab':      'sebab sebab',
    'caracara':        'cara cara',
}

# Pre-substitutions applied BEFORE BasicTextNormalizer strips punctuation.
# Use for patterns whose meaning depends on punctuation (e.g. H&M → hnm).
_PRE_SUBS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'\bH&M\b', re.IGNORECASE), 'hnm'),
]

# Phrase-level substitutions applied before per-token lookup.
# Handles multi-word variants that cannot be caught by a single-token dict.
_PHRASE_SUBS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'\btak ada\b'),          'tiada'),
    (re.compile(r'\bapa kah\b'),          'apakah'),
    (re.compile(r'\bmulti national\b'),   'multinational'),
    (re.compile(r'\block down\b'),        'lockdown'),
    # numeric suffix -an written with a space: "30 an" → "30an"
    (re.compile(r'\b(\d+)\s+an\b'),       r'\1an'),
    # RM currency prefix: rm200 / rm 200 → "200 ringgit"
    # BasicTextNormalizer lowercases and strips commas, so RM1,000 → rm1000
    (re.compile(r'\brm\s*(\d+)\b'),       r'\1 ringgit'),
    # Malay fractions: "satu per tiga" ↔ "satu pertiga" — fuse the split form
    (re.compile(r'\bper\s+(dua|tiga|empat|lima|enam|tujuh|lapan|sembilan|sepuluh)\b'), r'per\1'),
    # wrong word-boundary split: "penyu dah" → "penyudah" (penyu=turtle, dah=already, but together = finisher)
    (re.compile(r'\bpenyu\s+dah\b'), 'penyudah'),
    # covid-19 variants: hyphen stripped by BasicTextNormalizer → "covid 19"; Malay spoken form = "covid sembilan belas"
    (re.compile(r'\bcovid\s+19\b'),                    'covid19'),
    (re.compile(r'\bcovid\s+sembilan\s+belas\b'),      'covid19'),
    # "dia orang" → "diorang" (they/them — split vs fused colloquial pronoun)
    (re.compile(r'\bdia\s+orang\b'),                   'diorang'),
    # "di pertua" → "dipertua" (title: Yang Dipertua — split vs fused)
    (re.compile(r'\bdi\s+pertua\b'),                   'dipertua'),
    # "apa hal" → "apahal" (split vs fused interrogative)
    (re.compile(r'\bapa\s+hal\b'),                     'apahal'),
    # "mak cik" → "makcik" (split vs fused)
    (re.compile(r'\bmak\s+cik\b'),                     'makcik'),
    # "mat sui" → "matsui" (name: split vs fused)
    (re.compile(r'\bmat\s+sui\b'),                     'matsui'),
    # "pak cik" already handled via pakcik → 'pak cik' canonical; also fuse split form
    (re.compile(r'\bpak\s+cik\b'),                     'pak cik'),
]

# Arabic numerals → Malay words (single digits + common round numbers seen in data)
_DIGIT_WORDS: dict[str, str] = {
    '0': 'kosong',
    '1': 'satu',
    '2': 'dua',
    '3': 'tiga',
    '4': 'empat',
    '5': 'lima',
    '6': 'enam',
    '7': 'tujuh',
    '8': 'lapan',
    '9': 'sembilan',
    '10': 'sepuluh',
    '100': 'seratus',
}

# ke + digit ordinal expansion (ke3 → ketiga)
_ORDINAL_WORDS: dict[str, str] = {
    '1': 'satu', '2': 'dua',    '3': 'tiga',   '4': 'empat',
    '5': 'lima', '6': 'enam',   '7': 'tujuh',   '8': 'lapan',
    '9': 'sembilan', '10': 'sepuluh',
}
_KE_RE = re.compile(r'^ke(\d+)$')

# Malay sentence-final particles that speakers (and models) often fuse onto
# the preceding word without a space, e.g. "okaylah", "okeylah", "bolehlah".
# Ordered longest-first so "lah" doesn't shadow "leh" etc.
_PARTICLES: tuple[str, ...] = ('lah', 'leh', 'mah', 'kan', 'kot', 'pun', 'la', 'je')

# Only split fused particles off these bases — avoids false-positives on words
# that genuinely end in a particle string (e.g. "tulah" = curse word).
_SPLITTABLE: frozenset[str] = frozenset({
    # okay cluster (all variants + canonical)
    'okay', 'okey', 'oke', 'ok',
    # demonstratives
    'ni', 'ini',
    'tu', 'itu',
    # other high-frequency fused forms seen in data
    'juga', 'jugak',
    'ya', 'yeah',
    'boleh',
    'tak', 'tidak',
    'ada',
    'dah', 'sudah',
    'dulu',
    'jom',
    'nak', 'hendak',
    'sama',
    'cakap',
    'tempat',
    'tunjuk',
    # sekejap / kejap variants (e.g. sekejaplah, kejaplah → jap lah)
    'sekejap', 'kejap', 'jap',
    # compound bases
    'takyah',   # takyah = tak + yah (written as one word)
    'takya',    # takya  = tak + ya  (variant spelling)
    'takpe',    # takpe  = tak apa (never mind) — for takpelah etc.
    'takpa',    # takpa  = tak apa (variant spelling)
})

# Fused compound words that need splitting into multiple tokens.
# Applied before particle splitting so "takyahlah" → ["tak","ya","lah"].
_COMPOUND_SPLITS: dict[str, list[str]] = {
    'takyah': ['tak', 'ya'],
    'takya':  ['tak', 'ya'],
}

# Possessive/referential clitic suffixes fused onto the preceding word.
# e.g. perjalananku → perjalanan aku, rumahmu → rumah kau, namanya → nama dia
_CLITIC_MAP: dict[str, str] = {'ku': 'aku', 'mu': 'mu', 'nya': 'nya'}

# Words that genuinely end in a clitic string but are NOT clitic+base.
# Without this list they would be wrongly split (e.g. waktu → wakt aku).
_CLITIC_EXCEPTIONS: frozenset[str] = frozenset({
    'waktu', 'bangku', 'tongku', 'buku', 'suku', 'ruku', 'baku',
    'paku', 'laku', 'taku', 'manu', 'sanya', 'hanya', 'kenya',
    'anya', 'unya',
})

# Minimum base length to trigger clitic splitting — avoids false positives on
# short words (e.g. "ku" standalone, "mu" standalone) while catching real
# possessives like rumah+ku (5), hati+ku (4), nama+ku (4).
_CLITIC_MIN_BASE = 4


def _split_clitic(tok: str) -> list[str] | None:
    """Return [base, pronoun] if tok is base + fused possessive clitic."""
    if tok in _CLITIC_EXCEPTIONS:
        return None
    for suffix, pronoun in _CLITIC_MAP.items():
        if tok.endswith(suffix):
            base = tok[: -len(suffix)]
            if len(base) >= _CLITIC_MIN_BASE and base.isalpha():
                return [base, pronoun]
    return None


def _split_particle(tok: str) -> list[str] | None:
    """Return [*base_tokens, particle] if tok is a splittable base + fused particle."""
    for particle in _PARTICLES:
        if tok.endswith(particle) and len(tok) > len(particle) + 1:
            base = tok[: -len(particle)]
            if base in _SPLITTABLE:
                base_tokens = _COMPOUND_SPLITS.get(base) or [_CANONICAL.get(base, base)]
                return base_tokens + [particle]
    return None


def _canonicalize(tok: str) -> list[str]:
    # ke3 → ketiga etc.
    m = _KE_RE.match(tok)
    if m:
        num = m.group(1)
        return ['ke' + _ORDINAL_WORDS[num]] if num in _ORDINAL_WORDS else [tok]

    # Single-digit numeral → Malay word
    if tok in _DIGIT_WORDS:
        return [_DIGIT_WORDS[tok]]

    # Compound word split: takyah → ['tak', 'ya']
    if tok in _COMPOUND_SPLITS:
        return _COMPOUND_SPLITS[tok]

    # Fused possessive clitic: perjalananku → ['perjalanan', 'aku']
    clitic = _split_clitic(tok)
    if clitic:
        return clitic

    # Fused particle split: okaylah → ['ok', 'lah'], takyahlah → ['tak', 'ya', 'lah']
    split = _split_particle(tok)
    if split:
        return split

    canonical = _CANONICAL.get(tok, tok)
    # Handle canonical values that expand to multiple tokens (e.g. pakcik → 'pak cik')
    if ' ' in canonical:
        return canonical.split()
    return [canonical]


class MalayTextNormalizer(BasicTextNormalizer):
    """
    BasicTextNormalizer + Malay canonical-variant correction.

    After lowercasing and stripping punctuation, maps common spelling
    variants to a single canonical form so alternate valid spellings
    don't count as WER errors. Also splits fused sentence-final particles
    (e.g. okaylah → ok lah, okeylah → ok lah).

    Examples: okay/okey/oke → ok, okaylah/okeylah → ok lah,
              tu → itu, ni/nih → ini, jugak → juga,
              sekejap/kejap → jap, sekejaplah → jap lah,
              takyahlah/takyalah/tak yalah → tak ya lah,
              deklalasi → deklarasi, filem/film → filem, 3/tiga → tiga,
              tak ada → tiada, takde/takdak → tiada.
    """

    def __call__(self, text: str) -> str:
        # Pre-subs before BasicTextNormalizer strips punctuation
        for pattern, replacement in _PRE_SUBS:
            text = pattern.sub(replacement, text)
        text = super().__call__(text)
        # Phrase-level substitutions before per-token lookup
        for pattern, replacement in _PHRASE_SUBS:
            text = pattern.sub(replacement, text)
        tokens: list[str] = []
        for t in text.split():
            tokens.extend(_canonicalize(t))
        return ' '.join(tokens)
