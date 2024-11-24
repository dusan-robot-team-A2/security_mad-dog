import logo from './logo.svg';
import './App.css';
import React, { useEffect, useState } from "react";

function App() {
  const [systemData, setSystemData] = useState({
    "emergency_status": false,
    "amr_status": 0,
    "amr_positioin": [0, 0],
    "da_track_data": [],
  });

  useEffect(() => {
    // SSE 연결 설정
    const eventSource = new EventSource("/data");
    console.log("connect")

    // // 메시지 이벤트 수신
    // eventSource.onmessage = (event) => {
    //   const newMessage = JSON.parse(event.data);
    //   console.log(newMessage)
    //   setSystemData(newMessage);
    // };

    // // 오류 처리
    // eventSource.onerror = () => {
    //   console.error("Error with SSE connection");
    //   eventSource.close();
    // };

    eventSource.onmessage = function(event) {
      const jsonData = JSON.parse(event.data); // JSON 데이터 파싱
      console.log(jsonData)
  };

    // 컴포넌트 언마운트 시 연결 종료
    return () => {
      console.log("close")
      eventSource.close();
    };
  }, []);

  return (
    <div className="App">
      <h1>
        진돗개
      </h1>
      <img
        src="/video_feed"
        alt="Video Stream"
      ></img>
      <div>
        <p>Emergency Status: {systemData.emergency_status}</p>
        <p>AMR Status: {systemData.amr_status}</p>
        <p>AMR Position: {systemData.amr_position}</p>
        <p>Tracking Data: {systemData.da_track_data}</p>
      </div>
    </div>
  );
}

export default App;
