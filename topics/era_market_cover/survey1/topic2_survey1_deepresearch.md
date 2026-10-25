# **音楽分析と生成のための深層学習：時代、市場、およびスタイル手がかりの分離**

音楽情報検索（MIR）および深層学習の交差点において、大規模なデータセットと事前学習済みオーディオエンコーダの進化により、音楽のジャンル、時代、および地理的市場の分類精度は飛躍的に向上している。しかし、提案されている大学院の最終プロジェクト（音源からの楽曲のリリース時代およびリリース市場の分類、およびその生成への応用）を遂行する上で、モデルが学習している特徴表現が「真の音楽的・様式的進化（アレンジ、楽器編成、和声など）」であるか、あるいは「データセットのバイアスや録音・マスタリングのアーティファクト（言語、音圧、特定の機材特性など）」であるかを分離することは極めて重要な課題である。本報告書は、指定された要件に基づき、ステムごとの分類寄与度の分析、ショートカット学習の評価、およびメロディを保持した時代スタイル変換生成（カバー曲生成）に関する包括的な文献調査と、限られた計算資源および期間（3〜4週間、GPUアクセスは最初の約1週間のみ）内で実行可能な具体的なプロジェクト設計を提供する。

## **1\. 検証済みの関連先行研究 (Closest prior work, verified)**

音楽の時代予測、市場分類、ステム単位の分析、およびショートカット学習の各領域において、以下の文献が本プロジェクトの理論的基盤および直接的な競合手法となる。先行研究は、音楽分類において機械学習モデルがいかに容易に表層的なアーティファクト（ショートカット）に過剰適合するかを証明している。

| 文献 | 発行年 | 発表の場 | 実施内容と報告された主要な結果 | 実施されていないこと / 本プロジェクトとの関連 |
| :---- | :---- | :---- | :---- | :---- |
| Serrà et al.1 | 2012 | Scientific Reports | 1955年から2010年までの西洋ポピュラー音楽464,411曲を分析し、時代が下るにつれてピッチ遷移の多様性が減少し、音色が均質化し、平均音圧（ラウドネス）が上昇していることを統計的に証明した。 | ステムごとの分析や深層学習を用いた予測は行っていない。本プロジェクトにおける時代分類の「ショートカット（音圧やマスタリング）」の音響的根拠となる（背景知識）。 |
| Sturm4 | 2014 | JNMR / ISMIR | GTZANデータセットを分析し、アーティストの重複や同一録音の別バージョンがデータセット内に散在していること（アーティスト効果）を発見。音楽ジャンル認識の評価が実際には「特定のアーティストの認識」に過ぎないことを指摘した。 | 時代や市場の予測には触れていない。データセットの分割におけるリーク防止（Out-of-fold fusionの重要性）の論拠となる（背景知識）。 |
| Zhang et al.7 | 2026 | ISMIR | 音楽の美的評価モデルにおける「ジャンルに起因するショートカット学習」を特定。ポップスを過大評価し、ジャズやクラシックを過小評価するバイアスを、過去の偏差に基づく適応的な重み付けによって緩和した。 | ステム単位の分離を用いたショートカットの解明は行っていない。本プロジェクトの分析フェーズ（バイアス評価）に対する強力な競合研究である。 |
| Schedl et al.9 | 2010 | ISMIR | Webページのヒット数、Twitterのハッシュタグ（\#nowplaying）、Last.fmの再生回数などのメタデータを用いて、国・地域別のアーティストの人気度を推定するヒューリスティクスを提案した。 | オーディオ信号自体からの地理的市場の予測や生成は行っていない。市場（マーケット）というラベルの定義とモデリングの歴史的背景を提供する（背景知識）。 |
| Du et al. (ByteCover3)10 | 2023 | ICASSP | CQT（定Q変換）チャンクから局所特徴を抽出し、MaxMean類似度とLocal Alignment Loss（LAL）を用いることで、短いクエリでも高精度なカバー曲識別（CSI）を達成した。 | カバー曲の生成自体は行っていない。生成されたカバー曲が元のメロディを保持しているかを客観的に評価するための指標として極めて重要である（競合・評価指標）。 |
| Copet et al. (MusicGen)12 | 2023 | arXiv / Meta | テキストプロンプトとクロマグラム（メロディ）を条件付けとして受け取り、単一ステージのTransformerと言語モデルを用いて高音質な音楽を生成する1.5BパラメータのMusicGen-Melodyを開発した。 | 分類器の分析やステムごとの解析は行っていない。本プロジェクトの生成デモにおいて直接利用すべき中核的なシステムである（競合・生成ツール）。 |
| Watcharasupat & Lerch14 | 2025 | arXiv | 潜在空間における領域クエリを用いて、音源分離をオーディオ・ツー・オーディオの検索タスクとして定式化した。 | 分離されたステムを用いた時代や市場の分類器へのプロービング（凍結特徴量の解析）は行っていない。ステムを用いた分析の最新動向を示す（背景知識）。 |
| Wu et al. (CLaMP 3\)15 | 2025 | ACL | 対照学習を用い、楽譜、オーディオ、MIDI、および100言語のテキストを共通の潜在空間にマッピングする多言語・マルチモーダルMIRフレームワークを提案し、SOTAを達成した。 | オーディオ生成やステム分離への応用は行っていない。オーディオエンコーダの凍結特徴量抽出（特に言語によるショートカットの検証）において利用可能である（背景知識）。 |

上記の通り、時代や市場の予測に関して、深層学習を用いた完全なエンドツーエンドの分類器が学習する特徴の内部構造を、ステム単位（ボーカル対伴奏）で系統的に分解し、さらにその結果を条件付き生成モデルにフィードバックして検証した単一の研究は存在しない。したがって、分類器のショートカット学習（言語やマスタリング）をステム分離によって暴くというアプローチは、MIR分野における重要なギャップを埋めるものである。

## **2\. データセット (Datasets)**

本プロジェクトでは、楽曲の「リリース時代」および「リリース市場（国や地域）」のメタデータを含み、かつモデルが特定のアーティストやアルバムの音響的特徴に過剰適合する「アルバム効果」や「アーティスト効果」を防ぐために、厳密な分割が可能なデータセットが必要である。期間が2週間未満であることを考慮し、オーディオの入手可能性も極めて重要である。

### **主要なデータセット候補**

**Discogs-VI および Discogs-VI-YT** 編集メタデータに基づく世界最大のオープンソース・バージョン識別データセットである。Discogs-VIは全体で約348,000の作品（クリーク）にわたる1,900,000以上のバージョンをメタデータとして含み、ジャンル、スタイル、レコードレーベル、リリースフォーマット、リリース日、およびリリース国の情報が網羅されている16。このうち、Discogs-VI-YTサブセットは、公式のYouTubeミュージックアップロードにマッピングされた約493,000のバージョン（約98,000作品）のYouTube URLを提供する17。マッピングアルゴリズムは、動画のカテゴリが「Music」であること、継続時間が20分未満であることなどをチェックし、タイトルやアーティスト名の完全一致やヒューリスティクスを用いて構築されているため、ノイズが少ない16。

* **サイズ:** オーディオがリンクされているのは493k曲。  
* **オーディオの取得:** YouTubeからのスクレイピングが必須。特徴量のみの提供はない。  
* **ライセンス:** メタデータはパブリックドメインに近いが、オーディオ自体は各権利者に帰属する。  
* **リーク問題:** 同一クリーク内の重複バージョンは手動で保持・排除されているが16、アーティストの重複については交差検証時に注意が必要である。  
* **実現可能性 (2週間以内):** 低〜中。YouTubeのジオブロックやAPI制限、URLの無効化（リンク切れ）、および数十万曲のダウンロードと前処理にかかる時間を考慮すると、数日以内に学習パイプラインに乗せるのは極めてリスクが高い16。

**Free Music Archive (FMA) / FMA-Small** FMAデータセットは、クリエイティブ・コモンズ・ライセンスの下で公開されている音楽ファイルとメタデータのコレクションである。

* **サイズ:** 完全版は約106,000曲、小規模版（FMA-Small）は8,000曲を含む18。  
* **オーディオの取得:** MP3形式のフルオーディオが直接ダウンロード可能である。  
* **ライセンス:** クリエイティブ・コモンズ（再配布・研究利用が容易）。  
* **リーク問題:** 文献において、FMAデータセットの標準的な分割にはアーティストの重複やリリース日の偏りに関するリークが存在することが指摘されている19。  
* **実現可能性 (2週間以内):** 高。オーディオが手元に即座に用意できるため、特徴量抽出にすぐ移行できる。

**MagnaTagATune** 25,887曲を含み、時代、ムード、楽器などのタグが付与された標準的なデータセットである。MAESTやMERTなどの事前学習モデルのベンチマークとして頻繁に利用される20。

* **実現可能性:** 高いが、本プロジェクトの目的である「リリース市場（地理的情報）」のメタデータが不足しているため、主目的には合致しない。

### **推奨データセット**

* **プライマリ（主）データセット:** **FMA (Free Music Archive) データセット**。オーディオの即時取得が可能であるため、GPUのアクセス期限（約1週間）に間に合わせるためには最適である。ただし、分析フェーズの厳密性を確保するために、提供されている標準のTrain/Test分割を破棄し、メタデータを用いて「同一アーティスト」および「同一アルバム」が完全に異なる分割に割り当てられるよう、独自のグループ化交差検証（GroupKFold）を構築する必要がある19。  
* **フォールバック（予備）データセット:** **Discogs-VI-YTの階層化サンプリング・サブセット**。FMAの市場ラベルが不十分な場合、Discogs-VI-YTのメタデータから特定の時代（例: 1970年代、1990年代、2010年代）および特定の市場（例: US、UK、日本）に限定して10,000曲程度をサンプリングし、並列処理で1〜2日のうちにYouTubeからオーディオをスクレイピングする。この規模であれば時間内に処理可能である。

## **3\. 分析ツールとしての音源分離 (Source separation as an analysis tool)**

時代や市場の分類器が「何を」学習しているか（ボーカルの言語・歌唱法か、伴奏の楽器編成・録音スタイルか）を検証するためには、高品質な音源分離（Music Source Separation: MSS）が不可欠である。分離モデルの選択は、品質（アーティファクトの少なさ）と計算コストのトレードオフとなる。

### **標準的な分離モデルと使用可能性**

> 1. **HTDemucs (Hybrid Transformer Demucs)** Demucsの発展型であり、2021年のMusic Demixing Challengeで優勝したモデルの改良版である。従来のリカレントボトルネック層を自己注意機構（Self-Attention）に置き換えている22。  
   * **計算コスト:** RTX 4090を1基使用した場合、10曲（約33分）のオーディオを73秒で処理可能である23。パラメータ数は約81.74Mであり、推論速度とメモリ効率に優れている22。  
   * **利点:** 短時間で数千〜数万曲のデータセットをボーカルと伴奏に分離可能であり、本プロジェクトの時間的制約に完全に合致する。  
> 2. **BS-RoFormer / Mel-RoFormer** Sound Demixing Challenge 2023（SDX'23）で優勝した現在の最高峰（SOTA）モデルである。入力スペクトログラムを周波数軸に沿って重複しないサブバンドに分割し（Band-Split）、Rotary Position Embedding（RoPE）を用いたTransformerブロックで処理する24。  
   * **品質:** MUSDB18HQデータセットにおいて、追加データなしで平均SDR（信号対歪み比）9.80 dBという驚異的な分離品質を達成している24。後継のMel-RoFormerはメル尺度を利用し、さらにボーカルの平滑性を向上させている26。  
   * **計算コスト:** BS-RoFormerやMelRoFoの大型モデル（パラメータ数 228.20M等）は推論に多大なVRAMと時間を要する22。  
   * **利点と欠点:** アーティファクトは最小限だが、数万曲の処理を1台のGPUで数日内に行うには重すぎる24。  
> 3. **Spleeter** Deezerによる初期のTensorFlowベースのモデル。スペクトログラムをピクセル単位で処理するが29、位相のキャンセルや他楽器のボーカルへの漏れ（ブリード）などのアーティファクトが強いため、精密なプロービング分析には不適である30。

### **分離アーティファクトによる情報リークのリスクと先行研究**

音源分離を分類器の入力として使用する最大の**リスク**は、分離アルゴリズム自体が生成するアーティファクト（ノイズ、位相の乱れ、帯域幅の欠損）が、時代やフォーマット（モノラル対ステレオ）の情報をリークさせることである。たとえば、1960年代のモノラル録音や帯域が狭いアナログ録音をHTDemucsに通した場合と、2010年代のハイレゾリューションなステレオ録音を通した場合とでは、分離モデルの「エラーの性質」が異なる。分類器は音楽的な特徴ではなく、この「分離エラーの分布」をショートカットとして学習してしまう危険性がある32。  
文献上でも、分離の品質が後段の検出精度に影響を与え、音源分離が主要なボトルネックになることが指摘されている33。このリークを防ぐためのコントロール実験として、分析の際には「抽出したボーカル」と「抽出した伴奏」を再合成（単純加算）したリミックス・オーディオを作成し、それをベースラインとして用いる手法が推奨される。これにより、分離モデルによる処理のアーティファクトがすべての入力データに均等に適用され、アーティファクト自体を分類の手がかりにすることを防ぐことができる。

## **4\. 時代と市場の手がかりに関する既知の知見 (Era and market cues, what is known)**

分類器が依存する「真のスタイル」と「ショートカット」を区別するために、音楽の進化と音響的相関に関する既存の文献の知見を整理する。

### **時代の音響的相関とショートカット**

西洋のポピュラー音楽の進化を定量化したSerràらの記念碑的な研究1 によれば、1955年から2010年にかけて、音楽は以下のような明確な音響的変化を遂げている。

* **ピッチ遷移の制限:** 和声の進行やメロディの複雑さが減少し、より予測可能で制限されたピッチパターンの使用が増加している。  
* **音色の均質化 (Timbral Homogeneity):** 年を追うごとに、異なる楽曲間で使用される音色のパレットが似通ってきている。  
* **ラウドネスの上昇 (Loudness War):** ダイナミックレンジコンプレッサーやリミッターの過度な使用により、平均的な音圧（ラウドネス）が継続的に上昇している2。

深層学習を用いた時代予測（Release Year Prediction）モデルは、和声やリズムの様式を学習する代わりに、この「ラウドネス（音圧）」や「周波数帯域の広さ（アナログ録音のノイズフロア）」をショートカットとして利用する傾向が極めて高い34。これらは音楽的なスタイルではなく、単なる「プロダクションとマスタリングのアーティファクト」である。

### **市場の音響的相関と言語のバイアス**

リリース市場や地理的起源の分類においては、民族音楽学的な観点からは特定の音階（ペンタトニックやマカーム）や固有の楽器編成が特徴となるはずである。近年では、デュアルブランチ深層学習やマルチヘッド自己注意機構を用いて地理的起源を識別する試みがなされている36。 しかし、オーディオ分類器において最大のショートカットとなるのは「ボーカルの言語」である。モデルが「日本の市場」を予測する際、J-POP特有のコード進行を認識しているのではなく、単に音声信号から「日本語の音韻や音声的特徴」を識別している可能性が高い37。

### **提案されるコントロール実験**

言語と音楽的要素を分離するために、以下のコントロール実験を設計する。

> 1. **インストゥルメンタル・サブセットでの評価:** ボーカルを含まない伴奏（Accompaniment）ステムのみを入力として市場分類器を学習・評価する。ここで精度がランダムレベルに落ちる場合、元のモデルは純粋に言語識別器として機能していたことが証明される。  
> 2. **言語識別（LID）の共変量分析:** 分離したボーカルステムを、OpenAIのWhisperなどの高性能な音声認識・言語識別モデルに入力する37。Whisperによる言語識別の確信度（Accuracy）と、MERTやMAESTなどの凍結特徴量に基づく市場分類器の予測精度の間に強い相関がある場合、それは音楽スタイルではなく言語のショートカットに依存している証拠となる。

## **5\. カバー曲またはスタイル変換生成 (Cover-song or style-transfer generation)**

分析によって安定した時代スタイルの手がかり（例えば、80年代のシンセポップ特有の音色やドラムパターン）が確認された場合、元のメロディを保持したまま楽曲を別の時代スタイルに変換（カバー生成）することが求められる。本タスクに適した生成システムを以下に概観する。

### **生成システムと手法の比較**

| システム / アプローチ | 入力と保持される要素 | 変更される要素 | オープンソース / 推論の計算要件 |
| :---- | :---- | :---- | :---- |
| **MusicGen-Melody** (Meta)12 | **入力:** テキストプロンプト \+ 参照オーディオのクロマグラム。 **保持:** 基礎となるメロディと和声の構造。 | **変更:** 楽器編成、ジャンル、音色、リズム、録音スタイル（テキストで指定されたすべての要素）。 | **公開:** CC-BY-NC 4.012。 **計算:** 単一ステージのTransformer（1.5Bパラメータ）。RTX 3090/4090 1基で数秒〜数分で生成可能13。 |
| **Diff-SVC / so-vits-svc**39 | **入力:** 元のボーカルステム。 **保持:** 言語的内容、歌詞、ピッチ（F0）、メロディ。 | **変更:** ボーカルの音色（声質）のみ。 | **公開:** 利用可能。 **計算:** VITSベースや拡散モデル。単一GPUで高速だが、伴奏や時代スタイルの変換は**不可能**39。 |
| **STEMPHONIC**41 | **入力:** テキスト \+ 期待されるステムの活動パターン。 **保持:** 指定されたタイミングとリズム構造。 | **変更:** 全体的な音響と指定された楽器の音色。 | **公開:** 最新の研究であり、制限あり。 **計算:** Stable Audio Openに似た数十億パラメータのDiTアーキテクチャ41。 |

### **本プロジェクトにおける推奨**

プロジェクトの制約（数人の学生、数週間、限られたGPUアクセス）を考慮すると、**MusicGen-Melody**が唯一の現実的かつ強力な選択肢である。 MusicGenは、オーディオを平坦化して予測するのではなく、EnCodecを用いた遅延パターン（Delay Pattern）を採用することで、コードブックをずらしながら自己回帰的に予測を行う。これにより、30秒のオーディオ（50Hz）を6,000ステップではなく1,500ステップで生成可能であり、単一のGPU（RTX 3090/4090等）で極めて効率的に動作する42。 クロマグラムによる条件付けによって元の楽曲のメロディ軌跡が強制的に保持される一方、テキストプロンプト（例: "A 1960s motown soul record with analog tape hiss and brass section"）によって、時代のスタイルやマスタリングの特性が完全に上書きされる12。Diff-SVCのような歌声変換モデルは、ピッチと声質の変換には優れているが、伴奏の再生成や時代特有のジャンル変換ができないため、本要件には適合しない39。

## **6\. 生成されたカバー曲の評価 (Evaluation of generated covers)**

生成されたカバー曲が「元のメロディを保持しているか」および「指定した時代スタイルを反映しているか」を主観的なリスニングテストのみで評価することは客観性に欠ける。以下の定量的なプロトコルを推奨する。

### **1\. スタイルの存在と音質の評価: Fréchet Audio Distance (FAD)**

FADは、参照となるデータセットと生成されたオーディオのセット間で、事前学習済みモデル（VGGishやCLAPなど）の潜在空間における多変量正規分布の平均と共分散の距離を計算する、リファレンスフリーの音質評価指標である44。

* **実装:** Microsoftの fadtk ライブラリを使用する。  
* **リファレンス設定:** MusicCapsのようなノイズの多い汎用データセットではなく、高品質な音楽としての分布を比較するために、**FMA-Popサブセット**を参照分布として使用することが推奨されている46。

### **2\. メロディの保持とカバー曲の同一性: ByteCover3**

単なるテキストからの音楽生成ではなく、元の曲の「カバー」として成立しているかを数学的に証明するために、最先端のカバー曲識別（Cover Song Identification: CSI）システムである**ByteCover3**を使用する10。

* **メカニズム:** ByteCover3は、オーディオからCQT（定Q変換）チャンクを抽出し、ResNet-IBNを通して局所特徴の埋め込みを得る。その後、MaxMean類似度（![][image1]）とLocal Alignment Loss（LAL）を用いて、テンポや楽器編成が変わっても基礎となる音楽的フレーズが一致しているかを高精度に判定する10。  
* **評価方法:** 原曲と生成されたカバー曲をByteCover3に入力し、Mean Average Precision (mAP) や Mean Rank (MR1) などの順位付け指標を計算する47。これにより、「メロディが保持されているか」を客観的に証明できる。

### **3\. テキスト（時代）アライメント: CLAPスコア**

LAION-CLAPや最新のCLaMP 3（多言語・マルチモーダル対照学習モデル）を用いて、プロンプトとして与えた時代のテキスト記述と、生成されたオーディオ間のコサイン類似度を計算し、スタイル指定が守られているかを確認する15。

### **循環論法（Circularity Risk）の回避**

分析フェーズで使用した分類器（例：MERTの特徴量で学習した時代分類器49）を、生成されたカバー曲の「時代」の評価者としてそのまま使用することは危険である。生成モデル（MusicGen）が、MERTがショートカットとして依存する特定の周波数帯のノイズ（例：テープヒスやラウドネス）を模倣するだけで分類器を騙すことができるからである。評価は、上記のようなFADやCLAPなどの独立した指標に依存すべきである。

## **7\. 新規性チェック (Novelty check)**

文献調査の結果、本提案プロジェクトは高い新規性を有している。

* **ステム単位の分類による特徴の帰属（Feature Attribution）:** MERTやMAESTなどの深層音楽表現モデルを用いた自己教師あり学習のプロービング（分類）は数多く存在するが20、その入力として「フルミックス」「ボーカルのみ」「伴奏のみ」を系統的に比較し、時代分類や市場分類においてモデルが「声の言語」や「マスタリングの音圧」にどれだけショートカット依存しているかを統計的に明らかにした研究は極めて稀である。  
* **分類器から得た知見の生成による検証:** 分類器のバイアス分析にとどまらず、MusicGen-Melodyを用いたクロマ条件付き生成によって意図的に「時代の手がかり」を注入し、さらにByteCover3などのCSIシステムを用いて「メロディが保持されたカバー曲であること」を数学的に証明するパイプラインは、分析系MIRと生成系MIRを架橋する新しいアプローチである。

## **8\. 3つの具体的なプロジェクトバリエーション**

チームの計算資源と時間に合わせた3つのバリエーションを提案する。推奨は、GPUの期限切れリスクを最小化する「バリエーションA」である。

| 項目 | バリエーションA（分析重視・推奨） | バリエーションB（生成重視） | バリエーションC（新規性・バイアス緩和） |
| :---- | :---- | :---- | :---- |
| **リサーチクエスチョン** | モデルは時代と市場を分類する際、言語、アレンジ、マスタリングのどれに依存しているか？ | 安定したメロディ構造を維持したまま、音楽の時代スタイルをどこまで自由に変換できるか？ | ステムの欠落（Stem-dropout）を利用して、分類器のショートカット学習（言語依存など）を防げるか？ |
| **データ** | FMAデータセット（厳密にアーティストリークを防いだ交差検証分割）19。 | よく知られたポップス楽曲50曲の評価セット。 | Discogs-VI-YTの部分集合（数千曲）16。 |
| **モデル** | HTDemucs22, MERT49, MAEST20（凍結プローブ）, SVM/ロジスティック回帰。 | MusicGen-Melody13, ByteCover310, FADtk46。 | HTDemucs, MERT, Zhangらのバイアス緩和損失を用いたカスタム分類器7。 |
| **実験手順** | 1\. 楽曲をHTDemucsで分離。 2\. ミックス、ボーカル、伴奏のそれぞれからMERT等で特徴量抽出。 3\. 線形分類器で時代・市場を予測し、スコアのペアワイズ有意差検定を実施。 4\. MusicGenで少数のデモ生成。 | 1\. 50曲に対し、4つの異なる時代のプロンプトでMusicGen-Melodyを用いて生成。 2\. FADを用いてFMA-Pop分布との音質距離を計算。 3\. ByteCover3で原曲との類似度を測定。 | 1\. ベースライン分類器を学習。 2\. 学習中にランダムにボーカルや伴奏のステムをドロップダウンさせる手法で学習。 3\. 言語やマスタリングのショートカットへの依存度が下がったかを評価。 |
| **タイムアウト時の削減対象** | 特徴量抽出モデルをMERTの1つのみに絞る。 | ターゲットとする時代の数を4つから2つに減らす。 | 市場の分類を諦め、時代の分類タスクのみに集中する。 |
| **成果物** | ショートカット学習と特徴量の寄与に関する厳密な統計レポート。 | メロディ保持とスタイル変換を客観的指標（ByteCover3/FAD）で証明した生成カバー曲のポートフォリオ。 | 音楽タスクにおけるショートカット学習を緩和する新しい学習手法の提案。 |
| **主なリスク** | 音源分離モデル（HTDemucs）のアーティファクトが交絡変数となること。 | MusicGenの推論処理によってGPUの計算資源が枯渇し、期限に間に合わないこと。 | ステムを落とすことで情報量が減りすぎ、分類精度が崩壊して結論が出ないリスク。 |
| **推定作業量** | 12〜15人日（特徴量抽出後はCPUで一瞬で終わるためGPU切れに強い）。 | 10〜12人日（GPUでの推論と評価スクリプトの実行に依存）。 | 15〜20人日（カスタムデータローダーやハイパーパラメータの調整に時間がかかる）。 |

**推奨:** **バリエーションA**を選択すべきである。凍結された特徴量（Frozen embeddings）を扱うため、GPUが必要なのは最初のHTDemucsによる分離とMERT/MAESTの特徴量抽出のみである。一度特徴量をディスクに保存すれば、GPUアクセスが1週間で切れても、ノートPCのCPU上でロジスティック回帰やSVMの学習・交差検証・統計的有意差検定を心ゆくまで実行できる。コース要件である「生成トラック」を満たすために、少数の楽曲（例: 5曲）のみをMusicGen-Melodyで時代変換するデモを付加すれば、堅牢な分析と生成の両立が可能である。

## **9\. 必読文献リスト (Reading list)**

週末に準備を行う学生のために、理論、モデル構造、および評価手法を網羅する10本の必須論文を優先順位順に挙げる。

> 1. **Sturm (2014) \- "The State of the Art Ten Years After a State of the Art..."**6 音楽情報検索における「アーティスト効果」とデータセットの欠陥を暴き、厳密な分割がなぜ不可欠かを説く必読書。  
> 2. **Serrà et al. (2012) \- "Measuring the Evolution of Contemporary Western Popular Music"**1 過去50年間の音楽の音響的進化（ラウドネス・ウォー、ピッチの制限、音色の均質化）を定量化しており、「時代」とは何かを定義する基礎。  
> 3. **Zhang et al. (2026) \- "Genre Bias or Aesthetic Perception? Identifying and Mitigating Shortcut Learning in Music Evaluation"**7 音楽評価におけるジャンル起因のショートカット学習とデータセットバイアスを数学的制約によって緩和する最新の最先端研究。  
> 4. **Schedl et al. (2010) \- "Estimating Country-Specific Artist Popularity"**9 音楽の地理的・市場的メタデータがどのようにモデリングされてきたかを示す、市場分析の歴史的コンテキスト。  
> 5. **Araz et al. (2024) \- "Discogs-VI: A Musical Version Identification Dataset..."**16 バージョン識別とメタデータの宝庫であるDiscogs-VIデータセットの構築手法と、YouTubeオーディオマッピングの限界を詳述。  
> 6. **Rouard et al. (2022) / Défossez et al. (2021) \- "Hybrid Transformer Demucs" (関連文献)**22 プロジェクトで使用する音源分離モデル（HTDemucs）の速度と品質のトレードオフ、および自己注意機構の導入について理解する。  
> 7. **Li et al. (2023) \- "MERT: Acoustic Music Understanding Model..."**49 分析フェーズでプローブとして使用する主要な事前学習モデル（MERT）の、RVQ-VAEとCQT再構成に基づくアーキテクチャを解説。  
> 8. **Copet et al. (2023) \- "Simple and Controllable Music Generation"**12 MusicGen-Melodyの論文。EnCodecの遅延パターンとクロマグラムによるメロディ条件付けのメカニズムを理解するためのバイブル。  
> 9. **Du et al. (2023) \- "ByteCover3: Accurate Cover Song Identification on Short Queries"**10 生成されたカバー曲のメロディ構造が保持されているかを証明するために用いる、CQTチャンクとMaxMean指標の数学的背景。  
> 10. **Gui et al. / Microsoft (2024) \- "Adapting Frechet Audio Distance for Generative Music Evaluation"**45 生成されたオーディオの品質を客観的に評価するFAD（Fréchet Audio Distance）のベストプラクティスとFMA-Popの利用を提唱。

## **10\. 未検証または不確実な事項 (Unverified or uncertain)**

* **Discogs-VI-YTのスクレイピング速度の実態:** メタデータ上で493,000のYouTube URLが利用可能とされているが、YouTubeのレート制限（Rate-limiting）やボット対策の頻繁な更新により、2週間の枠内で大規模なオーディオスクレイピングが完了するかどうかは検証できていない16。  
* **最新モデル（MAEST / CLaMP 3）の正確なVRAM消費量:** MERTやMusicGen-MelodyについてはRTX 3090/4090 (24GB VRAM) での動作プロファイルが確立されているが13、最新のCLaMP 315 や MAEST20 における特徴量抽出時の最適なバッチサイズとVRAM溢れのリスクについては、経験的なテストが必要である。  
* **オーディオ圧縮がHTDemucsの分離に与える影響:** YouTube特有のOpus/AACなどの非可逆圧縮アーティファクトが、HTDemucsの位相再構成パイプラインとどのように相互作用し、時代に依存しない未知のノイズフロアを生み出すかについては、文献上での定量的な検証データが不足している。

#### **引用文献**

> 1. Evolution of Western Popular Music \| PDF \- Scribd, [https\://www\.scribd.com/document/383604394/2012-Measuring-the-Evolution-of-Contemporary-Western-Popular-Music-pdf](https://www.scribd.com/document/383604394/2012-Measuring-the-Evolution-of-Contemporary-Western-Popular-Music-pdf)  
> 2. Measuring the evolution of contemporary western popular music, [https\://pubmed.ncbi.nlm.nih.gov/22837813/](https://pubmed.ncbi.nlm.nih.gov/22837813/)  
> 3. Measuring the Evolution of Contemporary Western Popular Music, [https\://www\.researchgate.net/publication/225045716\_Measuring\_the\_Evolution\_of\_Contemporary\_Western\_Popular\_Music](https://www.researchgate.net/publication/225045716_Measuring_the_Evolution_of_Contemporary_Western_Popular_Music)  
> 4. The GTZAN dataset: Its contents, its faults, their effects on evaluation, [https\://arxiv.org/pdf/1306.1461](https://arxiv.org/pdf/1306.1461)  
> 5. The GTZAN dataset: Its contents, its faults, their effects on evaluation, [https\://arxiv.org/html/1306.1461v2](https://arxiv.org/html/1306.1461v2)  
> 6. Future Research in Music Information Retrieval, [https\://vbn.aau.dk/ws/files/185623960/SturmJNMR20131211.pdf](https://vbn.aau.dk/ws/files/185623960/SturmJNMR20131211.pdf)  
> 7. Genre Bias or Aesthetic Perception? Identifying and Mitigating, [https\://arxiv.org/pdf/2607.13903](https://arxiv.org/pdf/2607.13903)  
> 8. Genre Bias or Aesthetic Perception? Identifying and Mitigating, [https\://arxiv.org/abs/2607.13903](https://arxiv.org/abs/2607.13903)  
> 9. What's Hot? Estimating Country-specific Artist Popularity \- ISMIR 2010, [https\://ismir2010.ismir.net/proceedings/ismir2010-22.pdf](https://ismir2010.ismir.net/proceedings/ismir2010-22.pdf)  
> 10. ByteCover3: Accurate Cover Song Identification on Short Queries, [https\://www\.researchgate.net/publication/369413927\_ByteCover3\_Accurate\_Cover\_Song\_Identification\_on\_Short\_Queries](https://www.researchgate.net/publication/369413927_ByteCover3_Accurate_Cover_Song_Identification_on_Short_Queries)  
> 11. arXiv:2303.11692v1 \[cs.SD\] 21 Mar 2023, [https\://arxiv.org/pdf/2303.11692](https://arxiv.org/pdf/2303.11692)  
> 12. facebook/musicgen-melody \- Hugging Face, [https\://huggingface.co/facebook/musicgen-melody](https://huggingface.co/facebook/musicgen-melody)  
> 13. MusicMagus: Zero-Shot Text-to-Music Editing via Diffusion Models, [https\://arxiv.org/html/2402.06178v3](https://arxiv.org/html/2402.06178v3)  
> 14. arXiv:2501.16171v1 \[eess.AS\] 27 Jan 2025, [https\://arxiv.org/pdf/2501.16171](https://arxiv.org/pdf/2501.16171)  
> 15. arXiv:2502.10362v3 \[cs.SD\] 18 May 2025, [https\://arxiv.org/pdf/2502.10362](https://arxiv.org/pdf/2502.10362)  
> 16. Discogs-VI: A Musical Version Identification Dataset Based ... \- arXiv, [https\://arxiv.org/html/2410.17400v1](https://arxiv.org/html/2410.17400v1)  
> 17. (PDF) Discogs-VI: A Musical Version Identification Dataset Based on, [https\://www\.researchgate.net/publication/385177463\_Discogs-VI\_A\_Musical\_Version\_Identification\_Dataset\_Based\_on\_Public\_Editorial\_Metadata](https://www.researchgate.net/publication/385177463_Discogs-VI_A_Musical_Version_Identification_Dataset_Based_on_Public_Editorial_Metadata)  
> 18. Computational Intelligence in Pattern Recognition \- dokumen.pub, [https\://dokumen.pub/computational-intelligence-in-pattern-recognition-proceedings-of-cipr-2021-9811625425-9789811625428.html](https://dokumen.pub/computational-intelligence-in-pattern-recognition-proceedings-of-cipr-2021-9811625425-9789811625428.html)  
> 19. SACAIR 2020, [https\://sacair.org.za/wp-content/uploads/2021/01/SACAIR\_Proceedings-MainBook\_vFin\_sm.pdf](https://sacair.org.za/wp-content/uploads/2021/01/SACAIR_Proceedings-MainBook_vFin_sm.pdf)  
> 20. OMAR-RQ: Open Music Audio Representation Model Trained ... \- arXiv, [https\://arxiv.org/html/2507.03482v1](https://arxiv.org/html/2507.03482v1)  
> 21. Benchmarking Music Autotagging with MGPHot Expert Annotations, [https\://arxiv.org/pdf/2509.06936](https://arxiv.org/pdf/2509.06936)  
> 22. (PDF) Towards Reliable Objective Evaluation Metrics for Generative, [https\://www\.researchgate.net/publication/393724568\_Towards\_Reliable\_Objective\_Evaluation\_Metrics\_for\_Generative\_Singing\_Voice\_Separation\_Models](https://www.researchgate.net/publication/393724568_Towards_Reliable_Objective_Evaluation_Metrics_for_Generative_Singing_Voice_Separation_Models)  
> 23. all-in-one-fix 2.0.4 on PyPI \- Libraries.io, [https\://libraries.io/pypi/all-in-one-fix](https://libraries.io/pypi/all-in-one-fix)  
> 24. arXiv:2309.02612v2 \[cs.SD\] 10 Sep 2023, [https\://arxiv.org/pdf/2309.02612](https://arxiv.org/pdf/2309.02612)  
> 25. Music Source Separation with Band-Split RoPE Transformer \- arXiv, [https\://arxiv.org/html/2309.02612v2](https://arxiv.org/html/2309.02612v2)  
> 26. arXiv:2310.01809v1 \[cs.SD\] 3 Oct 2023, [https\://arxiv.org/pdf/2310.01809](https://arxiv.org/pdf/2310.01809)  
> 27. arXiv:2409.04702v1 \[cs.SD\] 7 Sep 2024, [https\://arxiv.org/pdf/2409.04702](https://arxiv.org/pdf/2409.04702)  
> 28. Qwen3.8-27B at 160K context on a SINGLE RTX 4090 — 47–57 tok, [https\://www\.reddit.com/r/Qwen\_AI/comments/1vqzl5l/qwen3827b\_at\_160k\_context\_on\_a\_single\_rtx\_4090/](https://www.reddit.com/r/Qwen_AI/comments/1vqzl5l/qwen3827b_at_160k_context_on_a_single_rtx_4090/)  
> 29. Demucs vs Spleeter \- The Ultimate Guide \| BeatsToRapOn Blog, [https\://beatstorapon.com/blog/demucs-vs-spleeter-the-ultimate-guide/](https://beatstorapon.com/blog/demucs-vs-spleeter-the-ultimate-guide/)  
> 30. Music-Source-Separation-Training (MSST) \- arXiv, [https\://arxiv.org/pdf/2607.23395](https://arxiv.org/pdf/2607.23395)  
> 31. Music Source Separation via Stem Discovery \- arXiv, [https\://arxiv.org/html/2609.30912v1](https://arxiv.org/html/2609.30912v1)  
> 32. Music Source Separation via Stem Discovery \- arXiv, [https\://arxiv.org/pdf/2609.30912](https://arxiv.org/pdf/2609.30912)  
> 33. A Stem-Agnostic Approach to Hybrid AI Music Detection \- arXiv, [https\://arxiv.org/html/2609.26956v1](https://arxiv.org/html/2609.26956v1)  
> 34. Minimax Optimal Online Stochastic Learning for Sequences ... \- arXiv, [https\://arxiv.org/pdf/1904.09369](https://arxiv.org/pdf/1904.09369)  
> 35. Continuous-depth Deep Gaussian Processes \- arXiv, [https\://arxiv.org/html/2610.30890v1](https://arxiv.org/html/2610.30890v1)  
> 36. Multi-scale Signed Recurrence Plot Based Time Series, [https\://www\.researchgate.net/publication/355519365\_Multi-scale\_Signed\_Recurrence\_Plot\_Based\_Time\_Series\_Classification\_Using\_Inception\_Architectural\_Networks](https://www.researchgate.net/publication/355519365_Multi-scale_Signed_Recurrence_Plot_Based_Time_Series_Classification_Using_Inception_Architectural_Networks)  
> 37. Scaling and Task Composition Effects in Whisper Adaptation \- arXiv, [https\://arxiv.org/html/2609.11302v1](https://arxiv.org/html/2609.11302v1)  
> 38. arXiv:2502.20583v1 \[cs.LG\] 27 Feb 2025 \- OpenReview, [https\://openreview.net/pdf?id=ew2kgotC9o](https://openreview.net/pdf?id=ew2kgotC9o)  
> 39. arXiv:2310.05118v1 \[cs.SD\] 8 Oct 2023, [https\://arxiv.org/pdf/2310.05118](https://arxiv.org/pdf/2310.05118)  
> 40. Towards Unified Song Generation and Singing Voice Conversion, [https\://arxiv.org/pdf/2606.07015](https://arxiv.org/pdf/2606.07015)  
> 41. Stemphonic: All-at-once Flexible Multi-stem Music Generation, [https\://www\.alphaxiv.org/abs/2602.09891](https://www.alphaxiv.org/abs/2602.09891)  
> 42. Simple and Controllable Music Generation – Read with an AI tutor, [https\://www\.chapterpal.com/s/bw8w54gl/simple-and-controllable-music-generation](https://www.chapterpal.com/s/bw8w54gl/simple-and-controllable-music-generation)  
> 43. Paper page \- Simple and Controllable Music Generation, [https\://huggingface.co/papers/2306.05284](https://huggingface.co/papers/2306.05284)  
> 44. Evaluation of Fréchet Audio Distance with Bass Sounds, [https\://sol.sbc.org.br/index.php/sbcm/article/download/43938/43701/](https://sol.sbc.org.br/index.php/sbcm/article/download/43938/43701/)  
> 45. Insights from Neural Audio Codec Embedding Distances \- arXiv, [https\://arxiv.org/pdf/2509.18823](https://arxiv.org/pdf/2509.18823)  
> 46. microsoft/fadtk: A simple library for Fréchet Audio Distance (FAD, [https\://github.com/microsoft/fadtk](https://github.com/microsoft/fadtk)  
> 47. arXiv:2010.14022v2 \[cs.SD\] 23 Apr 2021, [https\://arxiv.org/pdf/2010.14022](https://arxiv.org/pdf/2010.14022)  
> 48. Heterogeneous Contrastive Learning for Foundation Models ... \- arXiv, [https\://arxiv.org/pdf/2404.00225](https://arxiv.org/pdf/2404.00225)  
> 49. (PDF) MERT: Acoustic Music Understanding Model with Large, [https\://www\.researchgate.net/publication/371223274\_MERT\_Acoustic\_Music\_Understanding\_Model\_with\_Large-Scale\_Self-supervised\_Training](https://www.researchgate.net/publication/371223274_MERT_Acoustic_Music_Understanding_Model_with_Large-Scale_Self-supervised_Training)  
> 50. mert: acoustic music understanding model \- arXiv, [https\://arxiv.org/pdf/2306.00107](https://arxiv.org/pdf/2306.00107)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKcAAAAVCAYAAAAjFP6SAAAFIElEQVR4Xu2aeehtUxTHl3nMLBnikfEPZQiRMlNCSY+QRxFJ5qnIkKEMiShJkSlTlHnWQ0QRKfIHyjzP88z6/PZe7rrr7bPvub/u795fr/Op1T37u/bZZ5999tl77X2uSEdHR0dHx8LO0lEYM4tGIbBIFDrGwzdRGDMHqD0UxQnwbxQc+BaLYsfMcr/amVEcI6up/RnFCbGp2g5RzCwj9c7bMWJeVHs+imNmtj1w6rN5FDObyOyr70LLpKcqrn1LFCfMBVLvgPhOiWLHaDlP6g9hHMyXNF3ONmiXpgXQbTJD7bZTFMbIUpKmC2IsWF5tjZ77f7aJgmMttePUjo6OaUADvxbFwElqZ6vtGB2ZFdUuVFsuOjLLShplDszpzZwPBj1kOsi2Lr2E2lYuPQxbRqEC9To5ipktZHC9h+IotX/U9lB7Ru2SPu/Mw82Y7a72kaQHxq/d6JNq16s9mrV9s26gkYep8Mic3sj552bN7K2se+2xrJk+z6U9dDrzE2e9ndMGnYb2vEltdbUX1L52fnhX7dZ8zDbRO2rH9txT+DIj5Kc9zpCU7y61m9VOU/vL5WsD57+s9kdBfyloQL4fo+io1XsodpP+wjj+2aVLHC+pUUpG45wq6c06Ue0EtYPSaVXoVFw7Nqx1nEEaneEOl35cFswDW0vSL83pn5zPQ54NoyhpdsF3ltN4gPZg15Tk36/nnuKcrFsMy/EKPfcUvnNuLOX6Q9RJM2PYcfTXsLx/u2ODdBzNgYVizOup+YbidEmFfaW2TvCNG+pxdUGLN0tHjFrkWmnOw54hviujI7OkJH9pMUQ7NfngAylfl7ZFvyqn7b5eVTvYMjn2lHI5EPP7fE2xYBOH5l/KeMA7slbidmn2Qc03FOz8W0Nhv/W7xwrXv6ygxZv9vaDRWd7M+itqT+XjJtg7bPIzzeIrPehSfTxNfkZJ9O9ymj1D3+7xnL0KWgni2Tb5BkEZxPkGIV5TuYQPTT6o+aaFNQYW38zIBpKG+7a2XjptIFz74oIWb/bXoM3JaWI/4/KsNWHlrhwdGXylepfq42nys1hD/zzoLARt9vL1n5O1QRCetMlXgzaIZTB13xc04zlZML+n5mtNfMhA2i8kShBvnTuEseBqA9e+qKDFOjK6e62U5zqn2WhlfKu2vqRYOJ5noO8SRemNyHGFvrjaumr3SPKv0u+W/bN+SE6/7nxGrEtMGyxe9s7HMVZkBmHmGAbWEJ8FjTJZeZfwC9USNV9rKMSvIi0umgS2ILoi6GixTnFKLuVhYWWa//x3t6QOZJDHj1gGOttAJfDFlS2rf1vg4H/W+YAXiljZiPWFqMW0gf6wO/b53pcUMxs3SvLPd1rkGEkDlbG2NF8b8D0dRUft3NawDWJvHjapb8i/SHob35O0oOCFeVDtk6zR4LzZu0qaFkmjfyxp6wtukN59WMd5JB+zQieWpewPpX8k/VRSeeg/OJ0vM7VGtu0qjEVNZHvpvURcj7b2MALblIyVdkioV+nfQBZz259B3sjpUgdkwPlC6vcC30uvLvfm3ybwsUdb4hqpn9sxAmwBM0kIHc6P4jSp3YtfCIF10hJHSBrQmsA3L4odo4cRZ9Lftps6yTDw5ejOKGb4wuWvcXhOb+c0Dz72xkusJKOpb0cLbEtpkhCr7hPFIYkfNjwsniwMWlXS/dZGPh/6RAiNDotix8yxs6TYd5LQYUqxZxv4YsU2YQ0WalzjS0kdtAn73FqCGPyJKE6H/wAqAoaKXvfmYwAAAABJRU5ErkJggg==>