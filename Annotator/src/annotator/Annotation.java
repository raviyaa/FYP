/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package annotator;

/**
 *
 * @author Raviya
 */
public class Annotation {

    public String facial;
    public String body;
    public String voice;

    @Override
    public String toString() {
        return String.format("{facial:%s, body:%s, voice:%s}", facial, body, voice);
    }

    public String getFacial() {
        return facial;
    }

    public String getBody() {
        return body;
    }

    public String getVoice() {
        return voice;
    }

}
