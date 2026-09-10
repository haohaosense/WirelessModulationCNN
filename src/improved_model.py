import torch
import torch.nn as nn


class ImprovedModulationCNN(nn.Module):

    def __init__(self, num_classes=11, in_channels=2):
        super().__init__()

        self.features = nn.Sequential(

            # =========================
            # Block 1
            # =========================

            nn.Conv1d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            nn.Dropout(
                p=0.2
            ),


            # =========================
            # Block 2
            # =========================

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            nn.Dropout(
                p=0.2
            ),


            # =========================
            # Block 3
            # =========================

            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(
                output_size=1
            )
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                in_features=256,
                out_features=128
            ),

            nn.ReLU(),

            nn.Dropout(
                p=0.3
            ),

            nn.Linear(
                in_features=128,
                out_features=num_classes
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x