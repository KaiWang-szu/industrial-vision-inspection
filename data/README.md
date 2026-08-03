# Data contract

The course dataset is not redistributed. Place an authorized copy under this directory using the following layout:

```text
data/
├── measure/
│   └── <measurement images>
└── defect-detection/
    ├── product1/
    │   ├── OK/
    │   └── NG/
    │       ├── imagenormal/
    │       └── imagedrawn/
    └── product2/
        ├── OK/
        └── NG/
            ├── imagenormal/
            └── imagedrawn/
```

`imagenormal` contains source images. `imagedrawn` contains the corresponding LabelMe-style annotations used to form reference masks. Image extensions supported by the CLI include BMP, PNG, JPEG, and TIFF.
