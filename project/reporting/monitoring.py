import datetime
import pandas as pd
from sklearn import datasets

from evidently.metrics import (
    ColumnDriftMetric,
    ColumnSummaryMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric
    )


from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite

from evidently.ui.dashboards import (
    CounterAgg,
    DashboardPanelCounter,
    DashboardPanelPlot,
    PanelValue,
    PlotType,
    ReportFilter
    )


from evidently.ui.remote import RemoteWorkspace

from evidently.ui.workspace import (
    Workspace,
    WorkspaceBase
    )


from evidently.metric_preset import ClassificationPreset

from evidently.test_preset import (
    BinaryClassificationTestPreset,
    BinaryClassificationTopKTestPreset
    )


# Classification metrics
from evidently.metrics import (
    ClassificationQualityMetric,
    ClassificationClassBalance,
    ClassificationConfusionMatrix,
    ClassificationQualityByClass,
    ClassificationClassSeparationPlot,
    ClassificationProbDistribution,
    ClassificationRocCurve,
    ClassificationPRCurve,
    ClassificationPRTable,
    ClassificationLiftCurve,
    ClassificationLiftTable,
    ClassificationQualityByFeatureTable,
    ClassificationDummyMetric
    )

# Classification tests
from evidently.tests import ( 
    TestPrecisionScore,
    TestRecallScore,
    TestF1Score,
    TestAccuracyScore
    )




noms_colonnes = ["name"] + [f"data{i}" for i in range(1, 301)] + ["reference", "actual"]
df = pd.read_csv("../data/prod-data.csv", names=noms_colonnes, sep=";",header=None)
#print(df.isnull().sum())

# drop de actual
audio_ref = df.drop(columns=["actual"])
# drop de reference
audio_cur = df.drop(columns=["reference"])






#here 
# Add empty prediction column to ref_data for classification
ref_data_classification = audio_ref.copy()
ref_data_classification["actual"] = ref_data_classification["reference"]
prod_data_classification = audio_cur.copy()




WORKSPACE = "workspace"

YOUR_PROJECT_NAME = "Analyse appel audio"
YOUR_PROJECT_DESCRIPTION = "Analyse de l'audio"




# Data Drift report
def create_data_drift_report():
    data_drift_report = Report(
        metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            # *[ColumnDriftMetric(column_name=f"feature_{i}", stattest="wasserstein") for i in range(0, 40)],
            # *[ColumnSummaryMetric(column_name=f"feature_{i}") for i in range(0, 40)]
            ColumnDriftMetric(column_name=f"feature_0", stattest="wasserstein"),
            ColumnSummaryMetric(column_name=f"feature_0"),
            ColumnDriftMetric(column_name=f"feature_1", stattest="wasserstein"),
            ColumnSummaryMetric(column_name=f"feature_1")
        ],
        timestamp=datetime.datetime.now(),
        tags = ["Data Drift"]
    )
    #here
    print("Before Data Drift Report Run")
    print(audio_ref.describe())

    data_drift_report.run(reference_data=audio_ref, current_data=audio_cur)
    # data_drift_report.run(reference_data=ref_data, current_data=prod_data)
    #here
    print("After Data Drift Report Run")
    print(data_drift_report.stats())

    return data_drift_report



# Data Drift test suite
def create_data_drift_test_suite():
    data_drift_test_suite = TestSuite(
        tests=[DataDriftTestPreset()],
        timestamp=datetime.datetime.now(),
        tags = ["Data Drift"]
    )

    data_drift_test_suite.run(reference_data=audio_ref, current_data=audio_cur)
    # data_drift_test_suite.run(reference_data=ref_data, current_data=prod_data)
    return data_drift_test_suite




#here
# Classification Performance report
def create_classification_report():
    classification_report = Report(
        metrics=[
            ClassificationQualityMetric(),
            ClassificationClassBalance(),
            ClassificationConfusionMatrix(),
            ClassificationQualityByClass(),
            ClassificationDummyMetric(),
        ],
        timestamp=datetime.datetime.now(),
        tags = ["Classification Performance"]
    )

    
    classification_report.run(reference_data=ref_data_classification[["actual", "reference"]], current_data=prod_data_classification[["actual", "reference"]])
    #classification_report.run(reference_data=audio_ref, current_data=audio_cur)
    return classification_report

# Classification Performance test suite
def create_classification_test_suite():
    classification_test_suite = TestSuite(
        # tests=[BinaryClassificationTestPreset()],
        # tests=[BinaryClassificationTopKTestPreset(k=10, stattest='psi')],
        tests = [
            TestPrecisionScore(),
            TestRecallScore(),
            TestF1Score(),
            TestAccuracyScore(),
        ],
        timestamp=datetime.datetime.now(),
        tags = ["Classification Performance"]
    )

    
    classification_test_suite.run(reference_data=ref_data_classification[["actual", "reference"]], current_data=prod_data_classification[["actual", "reference"]])
    #classification_test_suite.run(reference_data=audio_ref, current_data=audio_cur)
    
    return classification_test_suite


# Create project's Dashboard
def create_project(workspace: WorkspaceBase):
    project = workspace.create_project(YOUR_PROJECT_NAME)
    project.description = YOUR_PROJECT_DESCRIPTION

    project.dashboard.add_panel(
        DashboardPanelCounter(
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            agg=CounterAgg.NONE,
            title="Audio Anomaly Detection Dataset",
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Model Calls",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="DatasetMissingValuesMetric",
                field_path=DatasetMissingValuesMetric.fields.current.number_of_rows,
                legend="count",
            ),
            text="count",
            agg=CounterAgg.SUM,
            size=1,
        )
    )

    # Classification Quality metrics
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="F1-score",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="ClassificationQualityMetric",
                field_path="current.f1",
            ),
            agg=CounterAgg.LAST,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Precision",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="ClassificationQualityMetric",
                field_path="current.precision",
            ),
            agg=CounterAgg.LAST,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Recall",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="ClassificationQualityMetric",
                field_path="current.recall",
            ),
            agg=CounterAgg.LAST,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Accuracy",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="ClassificationQualityMetric",
                field_path="current.accuracy",
            ),
            agg=CounterAgg.LAST,
            size=1,
        )
    )

    # Data Drift metrics
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Share of Drifted Features",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="DatasetDriftMetric",
                field_path="share_of_drifted_columns",
                legend="share",
            ),
            text="share",
            agg=CounterAgg.LAST,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Dataset Quality",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(metric_id="DatasetDriftMetric", field_path="share_of_drifted_columns", legend="Drift Share"),
                PanelValue(
                    metric_id="DatasetMissingValuesMetric",
                    field_path=DatasetMissingValuesMetric.fields.current.share_of_missing_values,
                    legend="Missing Values Share",
                ),
            ],
            plot_type=PlotType.LINE,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Features Drift Score",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    metric_args={"column_name.name": f"feature_{i}"},
                    field_path=ColumnDriftMetric.fields.drift_score,
                    legend=f"Feature_{i}",
                )
                for i in range (0, 40)
            ],
            plot_type=PlotType.BAR,
            size=1,
        )
    )

    project.save()
    return project


def create_final_project(workspace: str):
    ws = Workspace.create(workspace)
    project = create_project(ws)

    #Data drift Report and Test Suite
    data_drift_report = create_data_drift_report()
    ws.add_report(project.id, data_drift_report)

    data_drift_test_suite = create_data_drift_test_suite()
    ws.add_test_suite(project.id, data_drift_test_suite)


    #Classification Report and Test Suite
    classification_report = create_classification_report()
    ws.add_report(project.id, classification_report)

    classification_test_suite = create_classification_test_suite()
    ws.add_test_suite(project.id, classification_test_suite)

def create_demo_project(workspace: str):
    ws = Workspace.create(workspace)
    project = create_project(ws)

if __name__ == "__main__":
    create_demo_project(WORKSPACE)
    