import React, {useState, useEffect} from 'react';
import {SafeAreaView, StyleSheet, Text, View, TouchableOpacity, Image, ScrollView, ActivityIndicator} from 'react-native';
import {LinearGradient} from 'expo-linear-gradient';
import * as ImagePicker from 'expo-image-picker';
import * as tf from '@tensorflow/tfjs';
import { decodeJpeg, bundleResourceIO } from '@tensorflow/tfjs-react-native';
import * as blazeface from '@tensorflow-models/blazeface';
import * as FileSystem from 'expo-file-system';
import Svg, {Rect} from 'react-native-svg';
import * as VideoThumbnails from 'expo-video-thumbnails';

const App = () => {
	const [video, setVideo] = useState([]);
	const [pred, setPred] = useState("");
	const [frames, setFrames] = useState({});
  const [faceDetector, setFaceDetector] = useState("");
	const [loading, setLoading] = useState(false);
  const [deepFakeDetector, setDeepFakeDetector] = useState("");

  useEffect(() => {
    async function loadModel() {
			try {
				const tfReady = await tf.ready();
				// const modelJson = await require("./assets/model/model.json");
				// const modelWeight = await require("./assets/model/group1-shard.bin");
				// const dfDetector = await tf.loadLayersModel(bundleResourceIO(modelJson,modelWeight));
					const faceDetector =  await blazeface.load();
					setFaceDetector(faceDetector);
			} catch(err) {
				console.log(err);
			}
			// setDeepFakeDetector(maskDetector);
    }
    loadModel();
  }, []); 

  const getFaces = async() => {
    try {
			let tempArray = {};
			let images = [];
			if(video[0].type === "video") {
				const time = video[0].duration;
				for(let t = 0 ; t < time ; t = t + 1000) {
					const {uri} = await VideoThumbnails.getThumbnailAsync(video[0].uri, {time: t});
					images.push(uri);
				}
			}
			else {
				images.push(video[0].uri);
			}
			for(let ind = 0 ; ind < images.length ; ind++) {
				const uri = images[ind];
				const tfReady = await tf.ready();
				const imgB64 = await FileSystem.readAsStringAsync(uri, {
					encoding: FileSystem.EncodingType.Base64,
				});
				const imgBuffer = tf.util.encodeString(imgB64, 'base64').buffer;
				const raw = new Uint8Array(imgBuffer);
				const imageTensor = decodeJpeg(raw).resizeBilinear([128, 128]);
				const faces = await faceDetector.estimateFaces(imageTensor, false);
				for (let i = 0 ; i < faces.length ; i++) {
					// let width = parseInt((faces[i].bottomRight[1] - faces[i].topLeft[1]));
					// let height = parseInt((faces[i].bottomRight[0] - faces[i].topLeft[0]));
					// let faceTensor = imageTensor.slice([parseInt(faces[i].topLeft[1]), parseInt(faces[i].topLeft[0]), 0], [width,height, 3]);
					// faceTensor = faceTensor.resizeBilinear([128, 128]).reshape([1, 128, 128, 3]);
					if(!tempArray.hasOwnProperty(uri)) {
						tempArray[uri] = [];
					}
					tempArray[uri].push({
						id: uri + i, 
						location: faces[i]
					});
				}
			}
			setFrames(tempArray);
			setLoading(false);
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
							<ActivityIndicator visible={loading} textContent={'Loading...'} textStyle={{color: 'white', fontSize: '20'}}/>
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
					<View style={{...styles.listContainer, height: '20%'}}>
						<ScrollView style={{flexDirection: 'row'}} horizontal={true}>
							{Object.entries(frames).map(([key, value]) => {
								return (
									<View style={{flex: 1}} key={key}>
										<Image
											key={key}
											source={{
												uri: key,
											}}
											style={{
												width: 128,
												height: 128,
												resizeMode: 'contain'
											}}
										/>
										<Svg height={128} width={128} style={{marginTop: -128}}>
											{value.map((face) => {
												return (
													<Rect
														key={face.id}
														x={face.location.topLeft[0]}
														y={face.location.topLeft[1]}
														width={(face.location.bottomRight[0] - face.location.topLeft[0])}
														height={(face.location.bottomRight[1] - face.location.topLeft[1])}
														stroke='red'
														strokeWidth={1}
														fill=""
													/>
												);
											})}
										</Svg>
									</View>
								)
							})}
						</ScrollView>
					</View>
				)}
				{pred !== "" && (
					<View style={{}}>
						<Text style={styles.subHeading}>Video is {pred >= 0.5 ? "Real" : "Fake"}</Text>
					</View>
				)}
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
		height: '30%',
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
});

export default App;