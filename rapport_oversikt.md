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

---

## Modeller og resultater

| Modell | AUC | Accuracy | Precision | Recall | F1 | Treningstid | Epoker |
|---|---|---|---|---|---|---|---|
| **ResNet-50** | 0.9780 | 91.70% | 89.14% | 94.97% | 0.9196 | ~8.8 timer | 35 |
| **DenseNet-121** | 0.9863 | 94.12% | 93.46% | 94.87% | 0.9416 | ~11.7 timer | 32 |
| **ViT-B/16** | **0.9958** | **96.86%** | **98.02%** | **95.65%** | **0.9682** | ~24.3 timer | 17 |

---

## Modellbeskrivelser

### ResNet-50

CNN med residual connections (skip connections) som løser problemet med vanishing gradients i dype nettverk. 50 lag dypt. ResNet introduserte konseptet med å la input hoppe over ett eller flere lag, slik at nettverket kan lære identitetsfunksjoner og unngå degradering av ytelse ved økende dybde. Raskest å trene av de tre modellene, men lavest ytelse på denne oppgaven.

### DenseNet-121

CNN der hvert lag er koblet til alle påfølgende lag via dense connections. Dette gir bedre feature-gjenbruk, sterkere gradient-flyt og færre parametere enn tilsvarende ResNet-modeller. 121 lag dypt. DenseNet konkatenerer feature maps fra tidligere lag i stedet for å summere dem (som ResNet), noe som gir nettverket tilgang til et bredere sett med features. Middels ytelse og treningstid.

### ViT-B/16 (Vision Transformer)

Deler bildet i 16x16 patches og behandler dem som en sekvens av tokens, analogt med ord i NLP. Bruker self-attention (transformer-arkitektur) for å modellere globale sammenhenger mellom alle deler av bildet samtidig. Dette skiller seg fra CNN-er som primært fanger lokale mønstre gjennom konvolusjonskjerner. Klart best ytelse (96.86% accuracy, 99.58% AUC), men tredobbel treningstid sammenlignet med ResNet-50.

---

## Best og dårligst

### Best: ViT-B/16

Overlegen på alle metrikker. Spesielt sterk precision (98.02%), noe som betyr svært få falske positiver — kun 44 ekte bilder ble feilklassifisert som fake av 2 400. Konvergerte også raskest i antall epoker (17 epoker). Transformer-arkitekturens evne til å fange globale mønstre og sammenhenger i bildet ser ut til å være en klar fordel for å oppdage subtile AI-artefakter som kan være spredt over hele bildet.

### Dårligst: ResNet-50

91.70% accuracy og lavest AUC (0.978). Hadde spesielt svak precision (89.14%), med 261 ekte bilder feilklassifisert som fake. Likevel en solid baseline med god recall (94.97%), noe som betyr at den fanger opp de fleste AI-genererte bildene. ResNets lokale konvolusjonskjerner kan ha vanskeligheter med å fange globale inkonsistenser som kjennetegner AI-genererte bilder.

---

## Confusion Matrix (testsett, 4 800 bilder per modell)

### ResNet-50

| | Predikert Real | Predikert Fake |
|---|---|---|
| **Faktisk Real** | 2 139 | 261 |
| **Faktisk Fake** | 131 | 2 269 |

### ViT-B/16

| | Predikert Real | Predikert Fake |
|---|---|---|
| **Faktisk Real** | 2 356 | 44 |
| **Faktisk Fake** | 124 | 2 276 |

---

## Konklusjon

Alle tre modellene generaliserer godt, med testresultater konsistente med valideringsresultatene. ViT-B/16 er klart best egnet for oppgaven med å skille ekte bilder fra AI-genererte, med en trade-off i treningstid. DenseNet-121 er et godt mellomvalg dersom man har begrenset GPU-tid. Resultatene viser at transformer-baserte modeller har et fortrinn over tradisjonelle CNN-er for denne typen oppgave, trolig fordi de kan modellere globale sammenhenger i bildet som er viktige for å oppdage AI-artefakter.
