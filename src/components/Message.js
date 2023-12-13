import { Component } from "react";
class Message extends Component {
    render() {
        return <div>
        <h1>This is a Message class</h1>
            The data: {this.props.MessageComponentContent}
        </div>
    }
}

export default Message;