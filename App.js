import React, {useState} from 'react';
import {SafeAreaView, StyleSheet, Text, View, TouchableOpacity, Image, ScrollView, ActivityIndicator, Modal} from 'react-native';
import {LinearGradient} from 'expo-linear-gradient';
import * as ImagePicker from 'expo-image-picker';
import * as tf from '@tensorflow/tfjs';
import { decodeJpeg, bundleResourceIO } from '@tensorflow/tfjs-react-native';
import * as blazeface from '@tensorflow-models/blazeface';
import * as FileSystem from 'expo-file-system';
import Svg, {Rect} from 'react-native-svg';
import * as VideoThumbnails from 'expo-video-thumbnails';
import LottieView from 'lottie-react-native';
import * as ImageManipulator from 'expo-image-manipulator';
import * as MediaLibrary from 'expo-media-library';

const App = () => {
	const [video, setVideo] = useState([]);
	const [pred, setPred] = useState("");
	const [frames, setFrames] = useState({});
	const [loading, setLoading] = useState(false); 
	const [facePred, setFacePred] = useState({});
	const [selectedFrame, setSelectedFrame] = useState(null);
	const [pictureVisible, setPictureVisible] = useState(false);
	const stepArray = ['Loading Model...', 'Extracting frames...', 'Extracting faces...'];
	const [steps, setSteps] = useState(-1);

  const getFaces = async() => {
    try {
			const tfReady = await tf.ready();
			let tempArray = {};
			let images = [];
			let tempPred = {};
			setSteps(0);
			const faceDetector = await blazeface.load();
            console.log("face detector loaded");
			const modelJson = require('./assets/model/model.json');
			const modelWeight = await require('./assets/model/group1.bin');
			const dfDetector = await tf.loadLayersModel(bundleResourceIO(modelJson, modelWeight));
            console.log("df detector loaded");
			setSteps(1);
			if(video[0].type === "video") {
				const time = video[0].duration;
                let {status} = await MediaLibrary.requestPermissionsAsync();
                const options = {
                    sortBy: [MediaLibrary.SortBy.creationTime],
                    mediaType: [MediaLibrary.MediaType.video],
                };
                let { assets } = await MediaLibrary.getAssetsAsync(options);
                const newVideoUri = assets.find((x) => x.id === video[0].assetId)?.uri;
				for(let t = 0 ; t < time ; t = t + 1000) {
					const {uri} = await VideoThumbnails.getThumbnailAsync(newVideoUri, {time: t});
					images.push(uri);
				}
			}
			else {
				images.push(video[0].uri);
			}
            console.log("images extracted");
			let count = 0;
			let temp = 0.0;
			setSteps(2);
			for(let ind = 0 ; ind < images.length ; ind++) {
                let uri = images[ind];
                const fileInfo = await FileSystem.getInfoAsync(uri);
                console.log(fileInfo)
				// const imgB64 = await FileSystem.readAsStringAsync(uri, {
				// 	encoding: FileSystem.EncodingType.Base64,
				// });
                const manipResult = await ImageManipulator.manipulateAsync(
                    uri,
                    [],
                    {base64: true, compress: 1, format: ImageManipulator.SaveFormat.JPEG },
                  );
                const imgB64 = manipResult.base64;
                console.log(1);
				const imgBuffer = tf.util.encodeString(imgB64, 'base64').buffer;
                console.log(2);
				const raw = new Uint8Array(imgBuffer);
                console.log(3);
				const imageTensor = decodeJpeg(raw).resizeBilinear([256, 256]);
                console.log(4);
				const faces = await faceDetector.estimateFaces(imageTensor, false);
                console.log("faces extracted");
				count += faces.length;
				for (let i = 0 ; i < faces.length ; i++) {
					let width = parseInt((faces[i].bottomRight[1] - faces[i].topLeft[1]));
					let height = parseInt((faces[i].bottomRight[0] - faces[i].topLeft[0]));
					let faceTensor = imageTensor.slice([parseInt(faces[i].topLeft[1]), parseInt(faces[i].topLeft[0]), 0], [width,height, 3]);
					faceTensor = faceTensor.resizeBilinear([256, 256]).reshape([1, 256, 256, 3]);
					let result = await dfDetector.predict(faceTensor).data();
					temp += result[0];
					if(!tempArray.hasOwnProperty(uri)) {
						tempArray[uri] = [];
					}
					tempArray[uri].push({
						id: uri + i, 
						location: faces[i],
					});
					let id = uri + i;
					tempPred[id] = result[0];
				}
			}
            console.log(count)
			if(count > 0) {
				setPred(String(temp * 100 / count));
			}
			tf.disposeVariables();
			setFrames(tempArray);
			setLoading(false);
			setSteps(-1);
			setFacePred(tempPred);
    } catch(err) {
       	console.log("Unable to load image", err);
				setLoading(false);
    }
  };

	const handleChooseVideo = async () => {
		let permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (permissionResult.granted === false) {
            alert('Permission to access camera roll is required!');
            return;
        }
        let pickerResult = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.All,
            base64: true
        });
        if (pickerResult.cancelled === true) {
            return;
        }
        setVideo([...video, pickerResult]);
	};

	const runTest = () => {
		setLoading(true);
		getFaces();
	};

	const reset = () => {
		setVideo([]);
		setFrames([]);
		setPred("");
	}

	const openPictureModal = (id) => {
		setPictureVisible(true);
		setSelectedFrame(id);
	}

	const closePictureModal = () => {
		setPictureVisible(false);
		setSelectedFrame(null);
	}

  return (
		<SafeAreaView style={{flex: 1}}>
      <LinearGradient
        colors={['rgba(51, 51, 153, 1)', 'rgba(51, 51, 153, 0.5)']}
        style={styles.container}>
        <View style={styles.elementContainer}>
          <Text style={styles.heading}>Welcome to Faker</Text>
          <Text style={styles.subHeading}>Upload a Video to see if it's real</Text>
        </View>
				{video.length === 0 && (
					<TouchableOpacity style={styles.openButton} onPress={handleChooseVideo}>
						<Text style={styles.subHeading}>Upload</Text>
					</TouchableOpacity>
				)}
				{video.length !== 0 && (
					<View style={styles.listContainer}>
						<View style={styles.imageRow}>
							{video.map((image, index) => {
								return (
									<Image
										key={'key' + index}
										source={{
											uri: image.uri,
										}}
										style={{
											width: '100%',
											marginVertical: '11%',
											height: '95%',
											resizeMode: 'contain',
										}}
									/>
								);
							})}
						</View>
						{loading ? (
							<Modal 
								visible={loading}
								transparent
								animationType='fade'>
								<View style={styles.optionsModalOuterContainer}>
									<View style={{...styles.optionsModalInnerContainer,alignItems: 'center', justifyContent: 'center', borderTopLeftRadius: 25, borderTopRightRadius: 25, borderBottomLeftRadius: 25, borderBottomRightRadius: 25}}>
										<LottieView
											autoPlay
											style={{
												width: 256,
												height: undefined,
												aspectRatio: 1,
												backgroundColor: 'rgba(54, 54, 128, 1)',
											}}
											source={require('./assets/16432-scan-face.json')}
										/>
										<Text style={styles.subHeading}>{stepArray[steps]}</Text>
									</View>
								</View>
							</Modal>
						) : (
						<View style={styles.buttonContainer}>
							<TouchableOpacity style={{...styles.openButton, width: '40%'}} onPress={runTest}>
								<Text style={styles.subHeading}>Check</Text>
							</TouchableOpacity>
							<TouchableOpacity style={{...styles.openButton, width: '40%'}} onPress={reset}>
								<Text style={styles.subHeading}>Reset</Text>
							</TouchableOpacity>
						</View>)}
					</View>
				)}
				{!loading && Object.keys(frames).length !== 0 && (
					<View style={styles.listContainer}>
						<ScrollView style={{flexDirection: 'row'}} horizontal={true}>
							{Object.entries(frames).map(([key, value]) => {
								return (
									<View style={{flex: 1, alignItems: 'center', justifyContent: 'center', marginHorizontal: 4}} key={key}>
										<TouchableOpacity onPress={() => openPictureModal(key)}>
											<Image
												key={key}
												source={{
													uri: key,
												}}
												style={{
													width: 128,
													height: undefined,
													aspectRatio: 1,
													resizeMode: 'stretch'
												}}
											/>
											<Svg height={128} width={128} style={{marginTop: -128}}>
												{value.map((face) => {
													return (
														<Rect
															key={face.id}
															x={face.location.topLeft[0] / 2}
															y={face.location.topLeft[1] / 2}
															width={(face.location.bottomRight[0] - face.location.topLeft[0]) / 2}
															height={(face.location.bottomRight[1] - face.location.topLeft[1]) / 2}
															stroke={facePred[face.id] == 1 ? 'green' : 'red'}
															strokeWidth={1}
															fill=""
														/>
													);
												})}
											</Svg>
										</TouchableOpacity>
									</View>
								)
							})}
						</ScrollView>
					</View>
				)}
				{pred !== "" && !loading && (
					<View style={{}}>
						<Text style={styles.subHeading}>{video[0].type.toUpperCase()} is {parseFloat(pred).toFixed(2)}% Real</Text>
					</View>
				)}
				<Modal 
					visible={pictureVisible} 
					transparent
					animationType='slide'>
					<View style={styles.optionsModalOuterContainer}>
						<TouchableOpacity
							style={styles.closeBtn}
							onPress={closePictureModal}>
								<Text style={styles.subHeading}>Close</Text>
						</TouchableOpacity>
						<View style={styles.optionsModalInnerContainer}>
							<View style={{alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(54, 54, 128, 1)',}}>
								<Image
									key={selectedFrame}
									source={{
										uri: selectedFrame,
									}}
									style={{
										width: 256,
										height: undefined,
										aspectRatio: 1,
										resizeMode: 'stretch'
									}}
								/>
								<Svg height={256} width={256} style={{marginTop: -256}}>
									{selectedFrame !== null && frames[selectedFrame].map((face) => {
										return (
											<Rect
												key={face.id}
												x={face.location.topLeft[0]}
												y={face.location.topLeft[1]}
												width={(face.location.bottomRight[0] - face.location.topLeft[0])}
												height={(face.location.bottomRight[1] - face.location.topLeft[1])}
												stroke={facePred[face.id] == 1 ? 'green' : 'red'}
												strokeWidth={1}
												fill=""
											/>
										);
									})}
								</Svg>
							</View>
						</View>
						<View style={{width: '80%',alignItems: 'center',paddingVertical: 5,backgroundColor: 'rgba(54, 54, 128, 1)',borderBottomRightRadius: 50,borderBottomLeftRadius: 50,}}>
							<Text style={styles.subHeading}> </Text>
						</View>
					</View>
				</Modal>
			</LinearGradient>
		</SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
		flex: 1,
		alignItems: 'center',
		justifyContent: 'center',
	},

	elementContainer: {
		alignItems: 'center',
		justifyContent: 'center',
		width: '95%',
		height: '20%',
		paddingBottom: '1%',
	},

  heading: {
		fontSize: 40,
		fontWeight: 'bold',
		textAlign: 'center',
		color: 'white',
	},

  subHeading: {
		fontSize: 20,
		fontWeight: 'bold',
		textAlign: 'center',
		color: 'white',
	},

	openButton: {
		backgroundColor: '#80aaff',
		borderRadius: 10,
		height: 40,
		alignItems: 'center',
		justifyContent: 'center',
		elevation: 2,
		width: '50%',
	},

	listContainer: {
		alignItems: 'center',
		justifyContent: 'space-evenly',
		height: '25%',
		width: '95%',
		margin: '1%',
	},

	imageRow: {
		alignItems: 'center',
		justifyContent: 'center',
		width: '100%',
		height: '70%',
		margin: '1%',
	},

	buttonContainer: {
		flexDirection: 'row',
		alignItems: 'center',
		justifyContent: 'space-around',
		width: '90%',
	},

	predBox: {
		alignItems: 'center',
		justifyContent: 'center',
		height: '10%',
		width: '90%',
		borderWidth: 1,
	},

	optionsModalOuterContainer: {
    backgroundColor: 'rgba(0,0,0,0.7)',
    flex: 1,
    justifyContent: 'center',
		alignItems: 'center',
  },

  optionsModalInnerContainer: {
    backgroundColor: 'rgba(54, 54, 128, 1)',
    width: '80%',
    maxHeight: '70%',
		padding: 10
  },

  closeBtn: {
		width: '80%',
    alignItems: 'center',
    paddingVertical: 5,
    backgroundColor: 'rgba(54, 54, 128, 1)',
		borderTopRightRadius: 50,
		borderTopLeftRadius: 50,
  },
});

export default App;