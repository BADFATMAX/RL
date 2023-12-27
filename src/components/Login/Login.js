import React, {useState} from 'react';
import PropTypes from 'prop-types';

function get_token(user){
    return user["username"];
}

export default function Login({ setToken }) {
    const [username, setUserName] = useState();
    const handleSubmit = e => {
        const token = get_token({
          username,
        });
        setToken(token);
        sessionStorage.setItem("token", token)
      }
    
  return(
    <form onSubmit={handleSubmit}>
      <label>
        <p>Username</p>
        <input type="text" onChange={e => setUserName(e.target.value)} />
      </label>
      <div>
        <button type="submit">Submit</button>
      </div>
    </form>
  );
}

Login.propTypes = {
    setToken: PropTypes.func.isRequired
}
