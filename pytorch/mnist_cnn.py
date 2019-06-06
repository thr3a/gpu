import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, num_classes):
        """
        Convolutional Neural Network
        
        ネットワーク構成：
            input - CONV - CONV - MaxPool - CONV - CONV - MaxPool - FC - output
            ※MaxPoolの直後にバッチ正規化を実施

        引数：
            num_classes: 分類するクラス数（＝出力層のユニット数）
        """

        super(CNN, self).__init__() # nn.Moduleを継承する

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1), # 出力サイズ: チャネル=16, 高さ=27, 幅=27
            nn.BatchNorm2d(16)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1), # 出力サイズ: チャネル=32, 高さ=26, 幅=26
            nn.BatchNorm2d(32)
        )
        self.full_connection = nn.Sequential(
            nn.Linear(in_features=32*26*26, out_features=512), # in_featuresは直前の出力ユニット数
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=num_classes)
        )


    # Forward計算の定義
    # 参考：Define by Runの特徴（入力に合わせてForward計算を変更可）
    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)

        # 直前のMaxPoolの出力が2次元（×チャネル数）なので，全結合の入力形式に変換
        # 参考：KerasのFlatten()と同じような処理
        x = x.view(x.size(0), 32 * 26 * 26)

        y = self.full_connection(x)
        
        return y


# 学習用関数
def train(loader_train, model_obj, optimizer, loss_fn, device, total_epoch, epoch):
    
    model_obj.train() # モデルを学習モードに変更

    # ミニバッチごとに学習
    for data, targets in loader_train:

        data = data.to(device) # GPUを使用するため，to()で明示的に指定
        targets = targets.to(device) # 同上

        optimizer.zero_grad() # 勾配を初期化
        outputs = model_obj(data) # 順伝播の計算
        loss = loss_fn(outputs, targets) # 誤差を計算

        loss.backward() # 誤差を逆伝播させる
        optimizer.step() # 重みを更新する
    
    print ('Epoch [%d/%d], Loss: %.4f' % (epoch, total_epoch, loss.item()))


# テスト用関数
def test(loader_test, trained_model, device):

    trained_model.eval() # モデルを推論モードに変更
    correct = 0 # 正解率計算用の変数を宣言

    # ミニバッチごとに推論
    with torch.no_grad(): # 推論時には勾配は不要
        for data, targets in loader_test:

            data = data.to(device) #  GPUを使用するため，to()で明示的に指定
            targets = targets.to(device) # 同上

            outputs = trained_model(data) # 順伝播の計算

            # 推論結果の取得と正誤判定
            _, predicted = torch.max(outputs.data, 1) # 確率が最大のラベルを取得
            correct += predicted.eq(targets.data.view_as(predicted)).sum() # 正解ならば正解数をカウントアップ
    
    # 正解率を計算
    data_num = len(loader_test.dataset) # テストデータの総数
    print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(correct, data_num, 100. * correct / data_num))


def  main():

    # 1. GPUの設定（PyTorchでは明示的に指定する必要がある）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 2. ハイパーパラメータの設定（最低限の設定）
    batch_size = 100
    num_classes = 10
    epochs = 3

    # 3. MNISTのデータセットを取得
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original', data_home='./')

    # 4. データの設定（入力データは閉区間[0, 1]に正規化する）
    x = mnist.data / 255
    y = mnist.target

    # 5. DataLoaderの作成
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split

    # 5-1. データを学習用とテスト用に分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/7, random_state=0)

    # 5-2. データのフォーマットを変換：PyTorchでの形式 = [画像数，チャネル数，高さ，幅]
    x_train = x_train.reshape(60000, 1, 28, 28)
    x_test = x_test.reshape(10000, 1, 28 ,28)

    # 5-3. PyTorchのテンソルに変換
    x_train = torch.Tensor(x_train)
    x_test = torch.Tensor(x_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # 5-4. 入力（x）とラベル（y）を組み合わせて最終的なデータを作成
    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test, y_test)

    # 5-5. DataLoaderを作成
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # 6. モデル作成
    model = CNN(num_classes=num_classes).to(device)
    print(model) # ネットワークの詳細を確認用に表示

    # 7. 損失関数を定義
    loss_fn = nn.CrossEntropyLoss()

    # 8. 最適化手法を定義（ここでは例としてAdamを選択）
    from torch import optim
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 9. 学習（エポック終了時点ごとにテスト用データで評価）
    print('Begin train')
    for epoch in range(1, epochs+1):
        train(loader_train, model, optimizer, loss_fn, device, epochs, epoch)
        test(loader_test, model, device)


if __name__ == '__main__':
    main()