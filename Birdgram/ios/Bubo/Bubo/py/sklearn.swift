// sklearn: A very small subset of functionality ported from python

import Foundation
import Surge

// Avoid many-small-files bottleneck at mobile startup
//  - (See payloads.py for details)
public class OvRLogReg: sk.multiclass.OneVsRestClassifier<sk.linear_model.LogisticRegressionOneClass> {

  public convenience required init(load props: FileProps) throws {
    let timer = Timer()
    let _n_estimators  = try props.at("_n_estimators")  as Int
    let _coef_arr      = try props.at("_coef_arr")      as [[Float]]
    let _intercept_arr = try props.at("_intercept_arr") as [Float]
    self.init(
      estimators_: (0..<_n_estimators).map { i in
        sk.linear_model.LogisticRegressionOneClass(
          coef_:      _coef_arr[i],
          intercept_: _intercept_arr[i]
        )
      },
      multilabel_: try props.at("multilabel_")
    )
    _Log.info(String(format: "OvRLogReg.init: time[%.3f], props.path[%@]", timer.time(), props.path))
  }

}

// No analogue in py, since predict_proba is duck typed everywhere [I think]
//  - Protocols must be nested manually [https://stackoverflow.com/a/53233629/397334]
public protocol _Classifier {
  func predict_proba(_ X: Matrix<Float>) -> Matrix<Float>
}

public typealias sk = sklearn // Idiomatic shorthand
public enum sklearn {

  public enum base {

    // Protocols must be nested manually [https://stackoverflow.com/a/53233629/397334]
    public typealias Classifier = _Classifier

  }

  public enum decomposition {

    public class PCA: Loadable {

      public let whiten:              Bool
      public let mean_:               [Float]
      public let components_:         Matrix<Float>
      public let explained_variance_: [Float]

      public convenience required init(load props: FileProps) throws {
        let timer = Timer()
        self.init(
          whiten:              try props.at("whiten"),
          mean_:               try props.at("mean_"),
          components_:         try props.at("components_"),
          explained_variance_: try props.at("explained_variance_")
        )
        _Log.info(String(format: "PCA.init: time[%.3f], props.path[%@]", timer.time(), props.path))
      }

      public init(
        whiten:              Bool,
        mean_:               [Float],
        components_:         Matrix<Float>,
        explained_variance_: [Float]
      ) {
        self.whiten              = whiten
        self.mean_               = mean_
        self.components_         = components_
        self.explained_variance_ = explained_variance_
      }

      public func transform(_ _X: Matrix<Float>) -> Matrix<Float> {
        var X = _X
        X = X - np.broadcast_to(row: mean_, X.shape)
        X = X * components_.T
        if whiten {
          X = X ./ np.broadcast_to(row: sqrt(explained_variance_), X.shape)
        }
        return X
      }

    }

  }

  public enum linear_model {

    public class LogisticRegressionOneClass: sk.base.Classifier, Loadable {

      public let coef_:      [Float] // Specialized to Array i/o Matrix for one class (for simplicity)
      public let intercept_: Float   // Specialized to scalar i/o Array for one class (for simplicity)

      public convenience required init(load props: FileProps) throws {
        // let timer = Timer()
        self.init(
          coef_:      try (props.at("coef_")      as [[Float]]).only(), // Convert Matrix -> Array for one class
          intercept_: try (props.at("intercept_") as [Float]).only()    // Convert Array -> scalar for one class
        )
        // XXX Noisy (called n_sp times)
        // _Log.info(String(format: "LogisticRegressionOneClass.init: time[%.3f], props.path[%@]", timer.time(), props.path))
      }

      public init(
        coef_:      [Float],
        intercept_: Float
      ) {
        self.coef_      = coef_
        self.intercept_ = intercept_
      }

      public func predict_proba(_ X: Matrix<Float>) -> Matrix<Float> {
        let (n, _)   = X.shape
        let scores   = X * coef_ .+ np.broadcast_to(intercept_, n)
        let probs    = scipy.special.expit(scores)  // (n,) probs for our one class (positive class)
        let probsAll = Matrix([1 - probs, probs]).T // (n,2) probs for "all" classes (2 classes: negative + positive)
        precondition(probsAll.shape == (n, 2), "probsAll.shape[\(probsAll.shape)] == (n[\(n)], 2)")
        return probsAll
      }

    }

  }

  public enum multiclass {

    public class OneVsRestClassifier<E: sk.base.Classifier & Loadable>: sk.base.Classifier, Loadable {

      public let estimators_: [E]
      public let multilabel_: Bool

      public convenience required init(load props: FileProps) throws {
        let timer = Timer()
        self.init(
          estimators_: try props.at("estimators_"),
          multilabel_: try props.at("multilabel_")
        )
        _Log.info(String(format: "OneVsRestClassifier.init: time[%.3f], props.path[%@]", timer.time(), props.path))
      }

      public init(
        estimators_: [E],
        multilabel_: Bool
      ) {
        self.estimators_ = estimators_
        self.multilabel_ = multilabel_
      }

      public func predict_proba(_ X: Matrix<Float>) -> Matrix<Float> {
        let (n, _) = X.shape

        // Y[i,j] = Prob[sample i is class j]
        var Y = Matrix(estimators_.map { estimator -> [Float] in
          let Y = estimator.predict_proba(X)
          // Take (n,) positive class column from (n,2) matrix [negative class, positive class]
          precondition(Y.shape == (n, 2), "Y.shape[\(Y.shape)] == (n[\(n)], 2)")
          return Y[column: 1]
        }).T

        // If only one estimator, then we still want to return probabilities for two classes (negative, positive)
        //  - TODO Untested
        if estimators_.count == 1 {
          precondition(Y.shape == (n, 1), "Y.shape[\(Y.shape)] == (n[\(n)], 1)")
          let y = Y.grid
          Y = Matrix(columns: [1 - y, y])
        }

        // Normalize probs per sample to 1, unless multi-label
        if !multilabel_ {
          let rowProbs = np.sum(Y, axis: 1) // Rows are samples (columns are classes)
          Y = Y ./ np.broadcast_to(column: rowProbs, Y.shape)
        }

        return Y
      }

    }

  }

}
