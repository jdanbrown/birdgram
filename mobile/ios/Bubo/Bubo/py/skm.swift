// Like py sp14/skm.py

import Foundation
import Surge

public enum sp14 {
  public enum skm {
    public class SKM: Loadable {

      // Params (hyperparams)
      //  - Just the subset needed for prediction (.transform)
      public let normalize:   Bool
      public let standardize: Bool
      public let pca_whiten:  Bool
      public let do_pca:      Bool

      // Attributes (fitted params)
      //  - Just the subset needed for prediction (.transform)
      public let D:   Matrix<Float>
      public let pca: sk.decomposition.PCA

      public convenience required init(load props: FileProps) throws {
        let timer = Timer()
        self.init(
          normalize:   try props.at("normalize"),
          standardize: try props.at("standardize"),
          pca_whiten:  try props.at("pca_whiten"),
          do_pca:      try props.at("do_pca"),
          D:           try props.at("D"),
          pca:         try props.at("pca")
        )
        _Log.info(String(format: "SKM.init: time[%.3f], props.path[%@]", timer.time(), props.path))
      }

      public init(
        normalize:   Bool,
        standardize: Bool,
        pca_whiten:  Bool,
        do_pca:      Bool,
        D:           Matrix<Float>,
        pca:         sk.decomposition.PCA
      ) {
        self.normalize   = normalize
        self.standardize = standardize
        self.pca_whiten  = pca_whiten
        self.do_pca      = do_pca
        self.D           = D
        self.pca         = pca
      }

      public func transform(
        _ _X:    Matrix<Float>,
        rectify: Bool = false,
        nHot:    Int  = 0
      ) -> Matrix<Float> {
        var X = _X
        // Normalize data (per sample)
        if normalize   { preconditionFailure("Unimplemented: normalize (SKM._normalize_samples)") }
        // Standardize data (across samples)
        if standardize { preconditionFailure("Unimplemented: standardize (SKM._standardize_fit_transform)") }
        // PCA whiten
        if do_pca { X = pca.transform(X.T).T }
        // Dot product with learned dictionary
        X = X.T * D
        // Rectify
        if rectify  { preconditionFailure("Unimplemented: rectify") }
        // x-hot coding instead of just dot product
        if nHot > 0 { preconditionFailure("Unimplemented: nHot[\(nHot)] > 0") }
        return X.T
      }

    }
  }
}
