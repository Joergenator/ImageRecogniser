# Prosjektoversikt: Real vs AI-genererte bilder

## Oppgave

Binær klassifisering — skille ekte fotografier fra AI-genererte bilder.

**Datasett:** 48 000 treningsbilder (24 000 ekte + 24 000 fake) og 12 000 testbilder (6 000 + 6 000). Alle skalert til 224x224 piksler.

---

## Tilnærming

Transfer learning med ImageNet-forhåndstrente modeller og en to-fase treningsstrategi:

- **Fase 1 (Frozen):** 5 epoker med lr=1e-3, backbone frosset — kun klassifiseringshode trenes
- **Fase 2 (Fine-tuning):** Opp til 30 epoker med lr=1e-5, alle lag åpne

**Felles konfigurasjon:** Adam-optimizer, cosine annealing scheduler, BCEWithLogitsLoss, batch size 64, dropout 0.3, weight decay 1e-4, early stopping (patience=5).

**Augmentering:** Random horizontal flip, random resized crop (0.8–1.0), ImageNet-normalisering.

**Scratch-baseline:** I tillegg ble en variant av ResNet-50 trent fra scratch (uten ImageNet-vekter) med GELU-aktivering i stedet for ReLU, mild augmentering (colour jitter + random erasing), uten frozen-fase (lr=1e-5 hele veien) og uten label smoothing (patience=10). Denne tjener som baseline for å måle bidraget fra transfer learning.

---

## Modeller og resultater

| Modell | AUC | Accuracy | Precision | Recall | F1 | Treningstid | Epoker |
|---|---|---|---|---|---|---|---|
| **ResNet-50** | 0.9780 | 91.70% | 89.14% | 94.97% | 0.9196 | ~8.8 timer | 35 |
| **DenseNet-121** | 0.9863 | 94.12% | 93.46% | 94.87% | 0.9416 | ~11.7 timer | 32 |
| **ViT-B/16** | **0.9958** | **96.86%** | **98.02%** | **95.65%** | **0.9682** | ~24.3 timer | 17 |
| **ResNet-50 (scratch+GELU)** | 0.9234 | 84.40% | 81.87% | 88.37% | 0.8500 | ~27.7 timer | 79 |

---

## Modellbeskrivelser

### ResNet-50

CNN med residual connections (skip connections) som løser problemet med vanishing gradients i dype nettverk. 50 lag dypt. ResNet introduserte konseptet med å la input hoppe over ett eller flere lag, slik at nettverket kan lære identitetsfunksjoner og unngå degradering av ytelse ved økende dybde. Raskest å trene av transfer learning-modellene, men svakest ytelse av disse tre.

### DenseNet-121

CNN der hvert lag er koblet til alle påfølgende lag via dense connections. Dette gir bedre feature-gjenbruk, sterkere gradient-flyt og færre parametere enn tilsvarende ResNet-modeller. 121 lag dypt. DenseNet konkatenerer feature maps fra tidligere lag i stedet for å summere dem (som ResNet), noe som gir nettverket tilgang til et bredere sett med features. Middels ytelse og treningstid.

### ViT-B/16 (Vision Transformer)

Deler bildet i 16x16 patches og behandler dem som en sekvens av tokens, analogt med ord i NLP. Bruker self-attention (transformer-arkitektur) for å modellere globale sammenhenger mellom alle deler av bildet samtidig. Dette skiller seg fra CNN-er som primært fanger lokale mønstre gjennom konvolusjonskjerner. Klart best ytelse (96.86% accuracy, 99.58% AUC), men tredobbel treningstid sammenlignet med ResNet-50.

### ResNet-50 (scratch+GELU)

Samme arkitektur som ResNet-50 over, men trent fra tilfeldig initialisering (uten ImageNet-forhåndstrening) og med GELU i stedet for ReLU. GELU gir en glattere ikke-linearitet enn ReLU og brukes blant annet i moderne transformer-modeller. Trent uten frozen-fase, med mild augmentering og uten label smoothing. Tjener som baseline for hvor mye transfer learning bidrar — uten ImageNet-vektene faller AUC fra 0.9780 til 0.9234 på samme arkitektur, og treningstiden tredobles (79 epoker mot 35).

---

## Best og dårligst

### Best: ViT-B/16

Overlegen på alle metrikker. Spesielt sterk precision (98.02%), noe som betyr svært få falske positiver — kun 116 ekte bilder ble feilklassifisert som fake av 6 000. Konvergerte også raskest i antall epoker (17 epoker). Transformer-arkitekturens evne til å fange globale mønstre og sammenhenger i bildet ser ut til å være en klar fordel for å oppdage subtile AI-artefakter som kan være spredt over hele bildet.

### Dårligst: ResNet-50 (scratch+GELU)

Klart svakeste modell med 84.40% accuracy og AUC 0.9234. Trent fra scratch i 79 epoker over ~27.7 timer — uten ImageNet-vektene faller ytelsen ~7 prosentpoeng på accuracy sammenlignet med samme arkitektur trent som transfer learning. Dette illustrerer hvor mye lavnivå-features fra ImageNet bidrar selv på en oppgave (real vs AI) som er ganske ulik klassisk objektgjenkjenning. Av de tre transfer-baserte modellene var ResNet-50 (transfer) svakest med 91.70% accuracy og lavest precision (89.14%), trolig fordi ResNets lokale konvolusjonskjerner har vanskeligere for å fange globale inkonsistenser som kjennetegner AI-genererte bilder.

---

## Confusion Matrix (testsett, 12 000 bilder per modell)

### ResNet-50

| | Predikert Real | Predikert Fake |
|---|---|---|
| **Faktisk Real** | 5 308 | 692 |
| **Faktisk Fake** | 302 | 5 698 |

### DenseNet-121

| | Predikert Real | Predikert Fake |
|---|---|---|
| **Faktisk Real** | 5 602 | 398 |
| **Faktisk Fake** | 308 | 5 692 |

### ViT-B/16

| | Predikert Real | Predikert Fake |
|---|---|---|
| **Faktisk Real** | 5 884 | 116 |
| **Faktisk Fake** | 261 | 5 739 |

### ResNet-50 (scratch+GELU)

| | Predikert Real | Predikert Fake |
|---|---|---|
| **Faktisk Real** | 4 826 | 1 174 |
| **Faktisk Fake** | 698 | 5 302 |

---

## Konklusjon

Alle tre transfer learning-modellene generaliserer godt, med testresultater konsistente med valideringsresultatene. ViT-B/16 er klart best egnet for oppgaven med å skille ekte bilder fra AI-genererte, med en trade-off i treningstid. DenseNet-121 er et godt mellomvalg dersom man har begrenset GPU-tid. Resultatene viser at transformer-baserte modeller har et fortrinn over tradisjonelle CNN-er for denne typen oppgave, trolig fordi de kan modellere globale sammenhenger i bildet som er viktige for å oppdage AI-artefakter. Scratch-baselinen (ResNet-50 uten ImageNet-vekter) viser samtidig at transfer learning er en stor del av løftet — selv etter 79 epoker er den ~7 prosentpoeng bak samme arkitektur trent som transfer.
